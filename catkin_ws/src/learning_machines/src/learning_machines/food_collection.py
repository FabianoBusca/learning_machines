import cv2
import numpy as np
import os
import json
from datetime import datetime
from robobo_interface import IRobobo, SimulationRobobo

# Constants
BEST_GENOTYPE = [
    2.4627060210971528,
    0.22245210064434282,
    -0.23712053456020488,
    2.086472636449688,
    2.5172621168868288,
    0.33188530265494937,
    1.5333665331229962,
    1.6000456134688303,
    3.0,
    3.0,
    2.025441975716278,
    1.6583643668795025,
    2.129778681342938,
    0.8261431108179764,
    -0.6678169693643315,
    -2.8156985314751104,
    -2.0773239814262627,
    -2.481640944664592,
    1.1671979709955584,
    2.763355316408206,
    1.0637006843175236,
    0.6855034270068775,
    -1.2923092047344351,
    -2.214325031291805,
    -2.6467229245494446,
    0.7654473206965081,
    0.18960214361839878,
    0.9398480481869007,
    0.9563114208292456,
    0.35807696490921526,
    0.2958701371101502,
    -1.4490303058177312,
    -2.8239952441812775,
    -0.29789939102397067
  ]
GRID_SIZE = 0.2
LOG_PATH = "/root/results/logs/"


def make_log_dir(subdir, multiple_runs=False):
    base = os.path.join(LOG_PATH, subdir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(base, f"run_{timestamp}")
    if multiple_runs:
        path = f"{path}_{os.getpid()}"
    os.makedirs(path, exist_ok=True)
    return path


def save_log(log_dir, log_data, genotype):
    log_file = os.path.join(log_dir, "log.json")
    fitness, food_hits, avg_alignment, movement, visited, walls = calculate_fitness(log_data)
    data = {
        "genotype": genotype.tolist() if isinstance(genotype, np.ndarray) else genotype,
        "fitness": fitness,
        "food_hits": food_hits,
        "avg_alignment": avg_alignment,
        "movement": movement,
        "movement_reward": movement * 0.02,
        "visited_cells": visited,
        "wall_collisions": walls,
        "steps": log_data
    }
    try:
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Log saved to: {log_file}")
    except Exception as e:
        print(f"Error saving log: {e}")
    return data


def get_cell(position):
    if position is None:
        return None
    x, y = position
    return int(x // GRID_SIZE), int(y // GRID_SIZE)


def image_to_green_grid(image, grid_rows=3, grid_cols=3):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))

    h, w = green_mask.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    heatmap = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            cell = green_mask[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            mean_intensity = np.mean(cell) / 255.0
            heatmap.append(mean_intensity)

    return heatmap


def policy(irs, grid, genotype):
    ir_inputs = 1.0 - np.array(irs, dtype=np.float32) / 400.0
    ir_inputs = np.clip(ir_inputs, 0.0, 1.0)

    inputs = np.concatenate([ir_inputs, grid])  # shape: (17,)
    weights = np.array(genotype, dtype=np.float32).reshape((17, 2))

    action = np.dot(inputs, weights)
    left, right = np.clip(action * 100, -100, 100)
    return int(left), int(right)


def calculate_fitness(log_data, idx=None):
    if not log_data:
        return 0.0, 0, 0.0, 0.0, 0, 0

    food_hits = 0
    wall_collisions = 0
    alignment_sum = 0.0
    movement_sum = 0.0
    visited_cells = set()

    for step in log_data:
        movement = (step["left_speed"] + step["right_speed"]) / 2.0
        if movement < 0:
            movement = -movement / 2.0  # Reverse movement is half
        movement_sum += movement

        visited_cells.add(get_cell(step["position"]))

        irs = step["irs"]
        heatmap = step.get("heatmap", [])
        if not heatmap or len(irs) < 8:
            continue

        collision_threshold = 270

        # Group IR readings
        front_ir = max(irs[i] for i in [2, 3, 4, 5, 7])
        back_ir = max(irs[i] for i in [0, 1, 6])

        # Green heatmap groupings (grid index: 3x3)
        center_green = np.mean([heatmap[i] for i in [1, 4, 7]])
        alignment_sum += center_green

        # Detect and classify collision
        if back_ir > collision_threshold:
            wall_collisions += 1

        if front_ir > collision_threshold:
            if np.mean(heatmap) > 0:
                food_hits += 1
            else:
                wall_collisions += 1

    avg_alignment = alignment_sum / len(log_data)

    fitness = (
            food_hits * 100 +
            avg_alignment * 30 +
            movement_sum * 0.02 +
            len(visited_cells) * 6 -
            wall_collisions * 60
    )

    if idx is not None:
        print(
            f"[{idx}] fitness: {fitness:.2f}, food: {food_hits}, alignment: {avg_alignment * 30:.2f}, movement_reward: {movement_sum * 0.02:.2f}, exploration: {len(visited_cells)}, wall penalty: {wall_collisions}")

    return fitness, food_hits, avg_alignment, movement_sum, len(visited_cells), wall_collisions

def create_initial_population(pop_size=20, genome_size=34):
    return [np.random.uniform(-2.0, 2.0, genome_size) for _ in range(pop_size)]


def crossover(p1, p2):
    mask = np.random.rand(len(p1)) < 0.5
    child1 = np.where(mask, p1, p2)
    child2 = np.where(mask, p2, p1)
    return child1, child2


def mutate(genotype, strength=0.3):
    noise = np.random.normal(0, strength, size=genotype.shape)
    mutated = np.clip(genotype + noise, -3.0, 3.0)
    return mutated


def tournament_selection(population, scores, size=3, win_prob=0.8):
    selected = []
    for _ in range(len(population)):
        candidates = np.random.choice(len(population), size, replace=False)
        sorted_candidates = sorted(candidates, key=lambda i: -scores[i] if not np.isnan(scores[i]) else -float('inf'))
        winner = sorted_candidates[0] if np.random.rand() < win_prob else sorted_candidates[1]
        selected.append(population[winner].copy())
    return selected


def collect(rob: IRobobo, genotype=None, steps=30, delay_ms=500, log_run=True, multiple_runs=False):
    if genotype is None:
        genotype = BEST_GENOTYPE
    log_data = []

    if isinstance(rob, SimulationRobobo) and not rob.is_running():
        rob.play_simulation()
        rob.sleep(2)

    rob.set_phone_tilt_blocking(90, 100)
    rob.set_phone_pan_blocking(180, 100)

    for step in range(steps):
        try:
            irs = rob.read_irs()
            image = rob.read_image_front()
            heatmap = image_to_green_grid(image)

            cv2.imwrite(os.path.join(LOG_PATH, f"image_{step}.png"), image)
            # heatmap_array = np.array(heatmap).reshape((3, 3))
            # heatmap_image = (heatmap_array * 255).astype(np.uint8)
            # heatmap_image = cv2.resize(heatmap_image, (300, 300), interpolation=cv2.INTER_NEAREST)
            # heatmap_image = cv2.applyColorMap(heatmap_image, cv2.COLORMAP_JET)
            # cv2.imwrite(os.path.join(LOG_PATH, f"heatmap_{step}.png"), heatmap_image)

            if irs is None or len(irs) < 8:
                rob.sleep(0.2)
                continue
            left_speed, right_speed = policy(irs, heatmap, genotype)
            timestamp = datetime.now().isoformat()
            try:
                pos = rob.get_position()
                position = (round(pos.x, 2), round(pos.y, 2))
            except:
                position = None

            log_data.append({
                "timestamp": timestamp,
                "position": position,
                "irs": irs,
                "heatmap": heatmap,
                "left_speed": left_speed,
                "right_speed": right_speed
            })

            if log_run:
                print(f"[{step}] Position: {position}, Moving: L={left_speed}, R={right_speed}")

            rob.move_blocking(left_speed, right_speed, delay_ms)

        except Exception as e:
            print(f"[{step}] Error: {e}")
            rob.sleep(1)

    if isinstance(rob, SimulationRobobo):
        try:
            rob.stop_simulation()
        except Exception as e:
            print(f"Error stopping simulation: {e}")

    fitness, food_hits, avg_alignment, movement, visited, wall_collisions = calculate_fitness(log_data)

    result = {
        "genotype": genotype,
        "fitness": fitness,
        "food_hits": food_hits,
        "avg_alignment": avg_alignment,
        "movement": movement,
        "movement_reward": movement * 0.02,
        "visited_cells": visited,
        "wall_collisions": wall_collisions,
        "displacement": visited,
        "steps": log_data
    }

    if log_run:
        log_dir = make_log_dir("collect", multiple_runs)
        save_log(log_dir, log_data, genotype)

    return result

def evolve_eater_population(
    rob: IRobobo,
    generations=50,
    pop_size=20,
    elite_size=2,
    steps_per_episode=30,
    resume_from=None
):
    if resume_from:
        with open(os.path.join(resume_from, "evolution_log.json")) as f:
            previous_log = json.load(f)
        start_gen = len(previous_log["best_fitness_per_gen"])
        population = [np.array(g) for g in previous_log["final_population"]]
        log_dir = resume_from
        evolution_log = previous_log
        print(f"Resuming evolution from generation {start_gen + 1}")
    else:
        start_gen = 0
        population = create_initial_population(pop_size)
        log_dir = make_log_dir("food_evolution")
        evolution_log = {
            "best_fitness_per_gen": [],
            "avg_fitness_per_gen": [],
            "best_genotype": None,
            "parameters": {
                "generations": generations,
                "pop_size": pop_size,
                "elite_size": elite_size,
                "steps_per_episode": steps_per_episode
            }
        }
        print(f"Starting evolution with {generations} generations, population size {pop_size}")

    for g in range(start_gen, start_gen + generations):
        fitnesses, disps, penalties, gen_data = [], [], [], []

        for i, geno in enumerate(population):
            res = collect(rob, geno, steps=steps_per_episode, log_run=False)
            fitnesses.append(res["fitness"])
            disps.append(res["displacement"])
            penalties.append(res["wall_collisions"] * 60)

            gen_data.append({
                "individual": i + 1,
                "genotype": geno.tolist(),
                "fitness": res["fitness"],
                "food_hits": res["food_hits"],
                "avg_alignment": res["avg_alignment"],
                "movement_sum": res["movement"],
                "visited_cells": res["visited_cells"],
                "wall_collisions": res["wall_collisions"],
                "penalty": res["wall_collisions"] * 60,
                "displacement": res["displacement"]
            })

        best_fit = max(fitnesses)
        avg_fit = np.mean(fitnesses)
        best_idx = np.argmax(fitnesses)
        print(
            f"Gen {g + 1}: Best={best_fit:.2f}, Avg={avg_fit:.2f}, "
            f"Eater={max(disps)}, Penalty={penalties[np.argmax(disps)]:.2f}"
        )

        evolution_log["best_fitness_per_gen"].append(best_fit)
        evolution_log["avg_fitness_per_gen"].append(avg_fit)
        evolution_log["best_genotype"] = population[best_idx].tolist()

        with open(os.path.join(log_dir, f"generation_{g + 1:02d}_summary.json"), "w") as f:
            json.dump({
                "generation": g + 1,
                "individuals": gen_data,
                "best_fitness": best_fit,
                "avg_fitness": avg_fit
            }, f, indent=2)

        if g < start_gen + generations - 1:
            elites = [population[i].copy() for i in np.argsort(fitnesses)[-elite_size:]]
            parents = tournament_selection(population, fitnesses)
            offspring = []
            while len(elites) + len(offspring) < pop_size:
                p1, p2 = np.random.choice(len(parents), 2, replace=False)
                c1, c2 = crossover(parents[p1], parents[p2])
                offspring.extend([mutate(c1), mutate(c2)])
            population = elites + offspring[:pop_size - elite_size]

    # Save full population to allow resuming
    evolution_log["final_population"] = [geno.tolist() for geno in population]

    with open(os.path.join(log_dir, "evolution_log.json"), "w") as f:
        json.dump(evolution_log, f, indent=2)

    best_geno = np.array(evolution_log["best_genotype"])
    print(
        f"\nEvolution complete! Best fitness: {max(evolution_log['best_fitness_per_gen']):.2f}\n"
        f"Best genotype: {best_geno}"
    )
    return best_geno, evolution_log