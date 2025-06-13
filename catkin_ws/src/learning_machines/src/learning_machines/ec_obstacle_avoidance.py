import numpy as np
import os
import json
from datetime import datetime
from robobo_interface import IRobobo, SimulationRobobo

BEST_GENOTYPE = [
    -0.19554845598617623,
    -0.16383624376825423,
    2.0997243935594945,
    -0.9607173886354592,
    0.31378420950274044,
    -0.7689862143061483,
    -1.9097851410539537,
    -0.42729725935389995,
    -1.9853713684339425,
    1.5868500533519525,
    0.9055503896677263,
    1.9127277073873317,
    -1.6541008860778228,
    -0.03373806712368613,
    1.5795534081137808,
    -1.634336165772628
  ]

GRID_SIZE = 0.2
LOG_PATH = base="/root/results/logs/"

def make_log_dir(base, multiple_runs=False):
    base = os.path.join(LOG_PATH, base)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(base, f"run_{timestamp}")
    if multiple_runs:
        pid = os.getpid()
        path = f"{path}_{pid}"
    os.makedirs(path, exist_ok=True)
    return path


def save_log(log_dir, log_data, genotype, fitness):
    """Save genotype and steps to a single JSON file"""
    log_file = os.path.join(log_dir, "log.json")

    data = {
        "genotype": genotype.tolist() if isinstance(genotype, np.ndarray) else genotype,
        "fitness": fitness,
        "steps": log_data
    }

    try:
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Log saved to: {log_file}")
    except Exception as e:
        print(f"Error saving log: {e}")

    return data


def policy(irs, genotype):
    inputs = 1.0 - np.array(irs, dtype=np.float32) / 400.0
    inputs = np.clip(inputs, 0.0, 1.0)
    weights = np.array(genotype, dtype=np.float32).reshape((8, 2))
    action = np.dot(inputs, weights)
    left, right = np.clip(action * 100, -100, 100)
    return int(left), int(right)

def get_cell(position):
    """
    Get the cell coordinates based on the robot's position.
    The position is expected to be a tuple (x, y).
    """
    if position is None:
        return None
    x, y = position
    cell_x = int(x // GRID_SIZE)
    cell_y = int(y // GRID_SIZE)
    return cell_x, cell_y

def calculate_fitness(log_data, idx=None):
    """
    Calculate fitness based on robot performance
    Higher fitness = better performance
    """
    if not log_data:
        return 0.0

    total_distance = 0.0
    penalty = 0.0

    visited_cells = set()

    for step in log_data:
        forward_speed = (step["left_speed"] + step["right_speed"]) / 2.0
        total_distance += abs(forward_speed) * 0.02

        obstacle_risk = max(step["irs"])
        if obstacle_risk > 10000: # On simulation it seems to be 10-60k depending on the ray while on hardware it is 25000
            penalty += 100

        visited_cells.add(get_cell(step["position"]))

    displacement_reward = len(visited_cells) * 12

    fitness = total_distance + displacement_reward - penalty
    if idx:
        print(f"[{idx}] fitness: {fitness}, distance: {total_distance}, penalty: {penalty}, unique cells: {len(visited_cells)}")
    else:
        print(f"fitness: {fitness}, distance: {total_distance}, penalty: {penalty}, unique cells: {len(visited_cells)}")
    return fitness, penalty, len(visited_cells)


def create_initial_population(pop_size=20, genome_size=16):
    """Create initial population with random genotypes"""
    population = []
    for _ in range(pop_size):
        genotype = np.random.uniform(-2.0, 2.0, genome_size)
        population.append(genotype)
    return population


def crossover(parent1, parent2):
    """Create offspring using single-point crossover"""
    crossover_point = np.random.randint(1, len(parent1))

    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

    return child1, child2


def mutate(genotype, mutation_rate=0.1, mutation_strength=0.3):
    """Mutate a genotype with given rate and strength"""
    mutated = genotype.copy()

    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i] += np.random.normal(0, mutation_strength)
            mutated[i] = np.clip(mutated[i], -3.0, 3.0)

    return mutated


def tournament_selection(population, fitness_scores, tournament_size=3, win_prob=0.8):
    selected = []

    for _ in range(len(population)):
        indices = np.random.choice(len(population), tournament_size, replace=False)
        sub_pop = sorted(
            [(i, fitness_scores[i]) for i in indices],
            key=lambda x: -x[1] if not np.isnan(x[1]) else -float("inf")
        )
        if np.random.rand() < win_prob:
            selected.append(population[sub_pop[0][0]].copy())
        else:
            selected.append(population[sub_pop[1][0]].copy())

    return selected


def race(rob: IRobobo, genotype=None, steps=30, delay_ms=500, log_run=True, multiple_runs=False, idx=None):
    if genotype is None:
        genotype = BEST_GENOTYPE

    log_data = []

    if isinstance(rob, SimulationRobobo) and not rob.is_running():
        rob.play_simulation()
        rob.sleep(2)

    for step in range(steps):
        try:
            irs = rob.read_irs()
            if irs is None or len(irs) < 8:
                print(f"[{step}] Skipping due to invalid IR data: {irs}")
                rob.sleep(0.2)
                continue

            left_speed, right_speed = policy(irs, genotype)
            timestamp = datetime.now().isoformat()

            position = None
            try:
                p = rob.get_position()
                position = (round(p.x, 2), round(p.y, 2))
            except Exception as e:
                pass

            step_data = {
                "timestamp": timestamp,
                "position": position,
                "irs": irs,
                "left_speed": left_speed,
                "right_speed": right_speed
            }
            log_data.append(step_data)

            if log_run:
                print(f"[{step}] Position: {position}, Moving: L={left_speed}, R={right_speed}")

            rob.move_blocking(int(left_speed), int(right_speed), int(delay_ms))

        except Exception as e:
            print(f"[{step}] Error: {e}")
            rob.sleep(1)

    if isinstance(rob, SimulationRobobo):
        try:
            rob.stop_simulation()
        except Exception as e:
            print(f"Error stopping simulation: {e}")

    fitness, penalty, displacement = calculate_fitness(log_data, idx=idx)

    if log_run:
        log_dir = make_log_dir("race", multiple_runs=multiple_runs)
        race_log = save_log(log_dir, log_data, genotype, fitness)
        print(f"Race complete with fitness {fitness}, penalty {penalty}, displacement {displacement}. Logs saved at:", log_dir)
    else:
        race_log = {
            "genotype": genotype,
            "fitness": fitness,
            "penalty": penalty,
            "displacement": displacement,
            "steps": log_data
        }

    return race_log


def evolve_population(rob: IRobobo, generations=50, pop_size=20,
                      mutation_rate=0.1, elite_size=2, steps_per_race=30):
    """Main evolution loop with compact logging per generation"""

    print(f"Starting evolution with {generations} generations, population size {pop_size}")

    # Create initial population
    population = create_initial_population(pop_size)

    # Create one folder for the whole evolution run
    evolution_log_dir = make_log_dir("evolution")

    # Initialize evolution tracking structure
    evolution_log = {
        "best_fitness_per_gen": [],
        "avg_fitness_per_gen": [],
        "best_genotype": None,
        "parameters": {
            "generations": generations,
            "pop_size": pop_size,
            "mutation_rate": mutation_rate,
            "elite_size": elite_size,
            "steps_per_race": steps_per_race
        }
    }

    for generation in range(generations):
        fitness_scores = []
        displacements = []
        penalties = []
        generation_data = []

        for i, genotype in enumerate(population):
            # Run simulation without per-individual logging
            log_data = race(rob, genotype, steps=steps_per_race, log_run=False, idx=i)
            fitness = log_data["fitness"]
            penalty = log_data["penalty"]
            displacement = log_data["displacement"]
            fitness_scores.append(fitness)
            displacements.append(displacement)
            penalties.append(penalty)

            generation_data.append({
                "individual": i + 1,
                "genotype": genotype.tolist(),
                "fitness": fitness,
                "penalty": penalty,
                "displacement": displacement
            })

        # Compute generation stats
        best_fitness = max(fitness_scores)
        best_explorer = max(displacements)
        explorer_penalty = penalties[np.argmax(displacements)]
        avg_fitness = np.mean(fitness_scores)
        best_idx = np.argmax(fitness_scores)

        print(f"Generation {generation + 1} - Best: {best_fitness:.2f}, Avg: {avg_fitness:.2f} - Best Explorer: {best_explorer}, Penalty: {explorer_penalty:.2f}")

        # Append generation-wide stats
        evolution_log["best_fitness_per_gen"].append(best_fitness)
        evolution_log["avg_fitness_per_gen"].append(avg_fitness)
        evolution_log["best_genotype"] = population[best_idx].tolist()

        # Save generation summary to disk
        generation_summary = {
            "generation": generation + 1,
            "individuals": generation_data,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness
        }

        gen_filename = os.path.join(
            evolution_log_dir,
            f"generation_{generation + 1:02d}_summary.json"
        )
        try:
            with open(gen_filename, "w") as f:
                json.dump(generation_summary, f, indent=2)
            print(f"Saved generation summary: {gen_filename}")
        except Exception as e:
            print(f"Error saving generation summary: {e}")

        # Skip evolution step on last generation
        if generation < generations - 1:
            # Elite selection
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            new_population = [population[i].copy() for i in elite_indices]

            # Select parents
            parents = tournament_selection(population, fitness_scores)

            # Generate offspring
            while len(new_population) < pop_size:
                parent1, parent2 = np.random.choice(len(parents), 2, replace=False)
                child1, child2 = crossover(parents[parent1], parents[parent2])
                new_population.extend([
                    mutate(child1, mutation_rate),
                    mutate(child2, mutation_rate)
                ])

            population = new_population[:pop_size]

    # Save final evolution log
    evolution_log_file = os.path.join(evolution_log_dir, "evolution_log.json")
    try:
        with open(evolution_log_file, "w") as f:
            json.dump(evolution_log, f, indent=2)
        print(f"\nFinal evolution log saved to: {evolution_log_file}")
    except Exception as e:
        print(f"Error saving evolution log: {e}")

    best_genotype = np.array(evolution_log["best_genotype"])
    print(f"\nEvolution complete! Best fitness: {max(evolution_log['best_fitness_per_gen']):.2f}")
    print(f"Best genotype: {best_genotype}")

    return best_genotype, evolution_log

def explore(rob: IRobobo, cells=64, genotype=None, delay_ms=500):
    """
    Explore the environment until a certain number of cells are visited.
    """
    if genotype is None:
        genotype = BEST_GENOTYPE

    log_data = []

    if isinstance(rob, SimulationRobobo) and not rob.is_running():
        rob.play_simulation()
        rob.sleep(2)

    visited_cells = set()

    while len(visited_cells) < cells:
        try:
            irs = rob.read_irs()
            if irs is None or len(irs) < 8:
                print(f"Skipping due to invalid IR data: {irs}")
                rob.sleep(0.2)
                continue

            left_speed, right_speed = policy(irs, genotype)
            timestamp = datetime.now().isoformat()

            position = None
            try:
                p = rob.get_position()
                position = (round(p.x, 2), round(p.y, 2))
            except Exception as e:
                pass

            cell = get_cell(position)
            if cell is not None:
                visited_cells.add(cell)

            step_data = {
                "timestamp": timestamp,
                "position": position,
                "irs": irs,
                "left_speed": left_speed,
                "right_speed": right_speed,
                "visited_cells": len(visited_cells)
            }
            log_data.append(step_data)

            rob.move_blocking(int(left_speed), int(right_speed), int(delay_ms))

        except Exception as e:
            print(f"Error: {e}")
            rob.sleep(1)

        fitness, penalty, displacement = calculate_fitness(log_data)

        log_dir = make_log_dir("exploration", multiple_runs=False)
        save_log(log_dir, log_data, genotype, fitness)
        print(f"Race complete with fitness {fitness}, penalty {penalty}, displacement {displacement}. Logs saved at:", log_dir)

    if isinstance(rob, SimulationRobobo):
        try:
            rob.stop_simulation()
        except Exception as e:
            print(f"Error stopping simulation: {e}")