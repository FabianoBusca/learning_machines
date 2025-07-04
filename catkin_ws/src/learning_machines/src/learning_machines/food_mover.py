import cv2
import numpy as np
import os
import json
from datetime import datetime
from scipy.spatial.distance import cityblock
from robobo_interface import IRobobo, SimulationRobobo

# Constants
BEST_GENOTYPE = [
        -1.6405612486606769,
        1.5701654322720868,
        1.4604228668348376,
        -0.2680535133450599,
        0.21281612107711956,
        -0.43058988245946006,
        -0.9250173490616871,
        0.9839772284090582,
        -1.5843708052419592,
        1.304208137583725,
        1.3582834553382415,
        -1.9844970062426834,
        0.2650718472428544,
        -0.8343586294485497,
        1.9100132664852896,
        1.4143586183146946,
        -1.2285708434818714,
        -1.7826731263018991,
        -0.46620362737300214,
        0.41319389780044435,
        -1.3536617917449627,
        -1.256617290887526,
        1.6551547240717164,
        1.9167138600176075,
        1.1971628042437348,
        1.0114390506365614,
        -0.5009051778100608,
        1.525160249280881,
        0.21942674887133196,
        -1.319343465200204,
        -1.1200956426081863,
        -0.41703302817242927,
        -0.16439190535838932,
        -0.26946301855048205,
        -0.8346664306283134,
        -1.526739693684843,
        0.8945087768091238,
        -1.4388303546283026,
        1.748181566975994,
        0.4821975703698307,
        0.6549114903934821,
        -0.1718902412721537,
        0.24297081642030172,
        0.34739238630194036,
        1.9878006514434707,
        -1.577620240423677,
        -1.6755311493425218,
        0.06891479662667965,
        1.451260943404312,
        -1.6514873562632646,
        0.9057407283927699,
        -1.4172876148500824,
        -0.04823431994465155,
        1.2103952786566965,
        1.6677704623512328,
        1.2817841241919616
      ]
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
    fitness_result = calculate_fitness(log_data)
    rewards = fitness_result["rewards"]

    data = {
        "genotype": genotype.tolist() if isinstance(genotype, np.ndarray) else genotype,
        "fitness": fitness_result["fitness"],
        "success": fitness_result["success"],
        "red_alignment": fitness_result["red_alignment"],
        "green_alignment": fitness_result["green_alignment"],
        "movement": fitness_result["movement_sum"],
        "wall_collisions": fitness_result["wall_collisions"],
        "alignment_penalty_raw": fitness_result["alignment_penalty_raw"],
        **rewards,  # injects keys like red_alignment_reward, movement_reward, etc.
        "steps": log_data
    }

    try:
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Log saved to: {log_file}")
    except Exception as e:
        print(f"Error saving log: {e}")
    return data

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

def image_to_red_grid(image, grid_rows=3, grid_cols=3):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Red spans two ranges in HSV
    lower_red1 = (0, 100, 50)
    upper_red1 = (10, 255, 255)
    lower_red2 = (160, 100, 50)
    upper_red2 = (180, 255, 255)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    h, w = red_mask.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    heatmap = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            cell = red_mask[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            mean_intensity = np.mean(cell) / 255.0
            heatmap.append(mean_intensity)

    return heatmap


def policy(irs, red, green, genotype):
    ir_inputs = 1.0 - np.array(irs, dtype=np.float32) / 400.0
    ir_inputs = np.clip(ir_inputs, 0.0, 1.0)

    red = np.array(red, dtype=np.float32)
    green = np.array(green, dtype=np.float32)

    # Derived features
    alignment_score = np.array([red[7] * green[4]], dtype=np.float32)
    red_in_zone_feature = np.array([1.0 if red[7] > 0.1 and green[7] > 0.1 else 0.0], dtype=np.float32)

    # Final input vector: 8 + 9 + 9 + 1 + 1 = 28
    inputs = np.concatenate([ir_inputs, red, green, alignment_score, red_in_zone_feature])

    weights = np.array(genotype, dtype=np.float32).reshape((28, 2))

    action = np.dot(inputs, weights)
    left, right = np.clip(action * 100, -100, 100)
    return int(left), int(right)

def estimate_alignment_distance(red_heatmap, green_heatmap):
    red_idx = np.argmax(red_heatmap)
    green_idx = np.argmax(green_heatmap)
    red_coord = (red_idx // 3, red_idx % 3)
    green_coord = (green_idx // 3, green_idx % 3)
    return cityblock(red_coord, green_coord)

def calculate_fitness(log_data, idx=None):
    if not log_data:
        return {
            "fitness": 0.0,
            "rewards": {
                "red_alignment_reward": 0,
                "green_alignment_reward": 0,
                "both_alignment_reward": 0,
                "movement_reward": 0.0,
                "delivery_reward": 0,
                "wall_penalty": 0,
                "alignment_penalty": 0.0
            },
            "success": 0,
            "red_alignment": 0,
            "green_alignment": 0,
            "movement_sum": 0.0,
            "wall_collisions": 0,
            "alignment_penalty_raw": 0.0
        }

    wall_collisions = 0
    red_in_cell7_count = 0
    green_in_cell4_count = 0
    both_in_alignment = 0
    movement_sum = 0.0
    food_delivered = 0
    alignment_penalty_total = 0.0

    for step in log_data:
        irs = step.get("irs", [])
        red_heatmap = step.get("red_heatmap", [])
        green_heatmap = step.get("green_heatmap", [])
        red_in_green_zone = step.get("red_in_green_zone", False)

        if len(red_heatmap) != 9 or len(green_heatmap) != 9 or len(irs) < 8:
            continue

        movement = (step["left_speed"] + step["right_speed"])
        if movement < 0:
            movement = -movement / 2.0
        movement_sum += movement

        if red_heatmap[7] > 0.0:
            red_in_cell7_count += 1
        if green_heatmap[4] > 0.0:
            green_in_cell4_count += 1
        if red_heatmap[7] > 0.0 and green_heatmap[4] > 0.0:
            both_in_alignment += 1

        if red_in_green_zone:
            food_delivered += 1

        if max(irs) > 200:
            wall_collisions += 1

        alignment_penalty_total += estimate_alignment_distance(red_heatmap, green_heatmap)

    rewards = {
        "red_alignment_reward": red_in_cell7_count * 10,
        "green_alignment_reward": green_in_cell4_count * 10,
        "both_alignment_reward": both_in_alignment * 40,
        "movement_reward": movement_sum * 0.02,
        "delivery_reward": food_delivered * 2000,
        "wall_penalty": wall_collisions * 30,
        "alignment_penalty": alignment_penalty_total * 5
    }

    fitness = (
        rewards["red_alignment_reward"] +
        rewards["green_alignment_reward"] +
        rewards["both_alignment_reward"] +
        rewards["movement_reward"] +
        rewards["delivery_reward"] -
        rewards["wall_penalty"] -
        rewards["alignment_penalty"]
    )

    if idx is not None:
        print(
            f"[{idx}] fitness: {fitness:.2f} | red_7: {red_in_cell7_count}, green_4: {green_in_cell4_count}, align: {both_in_alignment}, "
            f"delivered: {food_delivered}, move: {movement_sum:.1f}, wall: {wall_collisions}, dist_penalty: {alignment_penalty_total:.2f}"
        )

    return {
        "fitness": fitness,
        "rewards": rewards,
        "success": food_delivered,
        "red_alignment": red_in_cell7_count,
        "green_alignment": green_in_cell4_count,
        "movement_sum": movement_sum,
        "wall_collisions": wall_collisions,
        "alignment_penalty_raw": alignment_penalty_total
    }
def create_initial_population(pop_size=20, genome_size=56):
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


def move(rob: IRobobo, genotype=None, steps=30, delay_ms=500, log_run=True, multiple_runs=False, hardware=False):
    if genotype is None:
        genotype = BEST_GENOTYPE
    log_data = []

    if isinstance(rob, SimulationRobobo) and not rob.is_running():
        rob.play_simulation()
        rob.sleep(2)

    if not hardware:
        rob.set_phone_tilt_blocking(90, 100)
        rob.set_phone_pan_blocking(180, 100)

    for step in range(steps):
        try:
            irs = rob.read_irs()
            image = rob.read_image_front()

            cv2.imwrite(os.path.join(LOG_PATH, f"image_{step}.png"), image)
            red_heatmap = image_to_red_grid(image)
            green_heatmap = image_to_green_grid(image)
            print(red_heatmap)
            print("---")
            print(green_heatmap)
            red_heatmap_img = (np.array(red_heatmap).reshape((3, 3)) * 255).astype(np.uint8)
            green_heatmap_img = (np.array(green_heatmap).reshape((3, 3)) * 255).astype(np.uint8)

            red_heatmap_img = cv2.resize(red_heatmap_img, (300, 300), interpolation=cv2.INTER_NEAREST)
            green_heatmap_img = cv2.resize(green_heatmap_img, (300, 300), interpolation=cv2.INTER_NEAREST)
            red_colored = cv2.applyColorMap(red_heatmap_img * 100, cv2.COLORMAP_HOT)
            green_colored = cv2.applyColorMap(green_heatmap_img * 200, cv2.COLORMAP_DEEPGREEN)
            cv2.imwrite(os.path.join(LOG_PATH, f"red_heatmap_{step}.png"), red_colored)
            cv2.imwrite(os.path.join(LOG_PATH, f"green_heatmap_{step}.png"), green_colored)
            if isinstance(rob, SimulationRobobo):
                try:
                    rob.stop_simulation()
                except Exception as e:
                    print(f"Error stopping simulation: {e}")
            exit(1)

            if irs is None or len(irs) < 8:
                rob.sleep(0.2)
                continue

            left_speed, right_speed = policy(irs, red_heatmap, green_heatmap, genotype)
            timestamp = datetime.now().isoformat()
            try:
                pos = rob.get_position()
                position = (round(pos.x, 2), round(pos.y, 2))
            except:
                position = None

            # red_in_green_zone = rob.base_detects_food()

            log_data.append({
                "timestamp": timestamp,
                "position": position,
                "irs": irs,
                "red_heatmap": red_heatmap,
                "green_heatmap": green_heatmap,
                "left_speed": left_speed,
                "right_speed": right_speed,
                "red_in_green_zone": False,
            })

            if log_run:
                print(f"[{step}] Position: {position}, Moving: L={left_speed}, R={right_speed}")

            if hardware:
                rob.move(left_speed, right_speed, delay_ms)
            else:
                rob.move_blocking(left_speed, right_speed, delay_ms)

        except Exception as e:
            print(f"[{step}] Error: {e}")
            rob.sleep(1)

    if isinstance(rob, SimulationRobobo):
        try:
            rob.stop_simulation()
        except Exception as e:
            print(f"Error stopping simulation: {e}")

    fitness_result = calculate_fitness(log_data)
    rewards = fitness_result["rewards"]

    result = {
        "genotype": genotype,
        "fitness": fitness_result["fitness"],
        "success": fitness_result["success"],
        "red_alignment": fitness_result["red_alignment"],
        "green_alignment": fitness_result["green_alignment"],
        "movement": fitness_result["movement_sum"],
        "wall_collisions": fitness_result["wall_collisions"],
        "alignment_penalty_raw": fitness_result["alignment_penalty_raw"],
        **rewards,
        "steps": log_data
    }

    if log_run:
        log_dir = make_log_dir("move", multiple_runs)
        save_log(log_dir, log_data, genotype)

    return result

def evolve_mover_population(
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
        log_dir = make_log_dir("move_evolution")
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
        fitnesses, gen_data = [], []

        for i, geno in enumerate(population):
            res = move(rob, geno, steps=steps_per_episode, log_run=False, hardware=False)
            fitnesses.append(res["fitness"])

            individual_data = {
                "individual": i + 1,
                "genotype": geno.tolist(),
                "fitness": res["fitness"],
                "success": res["success"],
                "green_alignment": res["green_alignment"],
                "red_alignment": res["red_alignment"],
                "movement_sum": res["movement"],
                "wall_collisions": res["wall_collisions"],
                "alignment_penalty_raw": res["alignment_penalty_raw"],
                "red_alignment_reward": res["red_alignment_reward"],
                "green_alignment_reward": res["green_alignment_reward"],
                "both_alignment_reward": res["both_alignment_reward"],
                "movement_reward": res["movement_reward"],
                "delivery_reward": res["delivery_reward"],
                "wall_penalty": res["wall_penalty"],
                "alignment_penalty": res["alignment_penalty"]
            }

            gen_data.append(individual_data)

        best_fit = max(fitnesses)
        avg_fit = np.mean(fitnesses)
        best_idx = np.argmax(fitnesses)
        print(
            f"Gen {g + 1}: Best={best_fit:.2f}, Avg={avg_fit:.2f}, "
            f"Move={gen_data[best_idx]['movement_reward']:.2f}, "
            f"Delivery={gen_data[best_idx]['delivery_reward']}, "
            f"Align={gen_data[best_idx]['both_alignment_reward']}, "
            f"Penalty={gen_data[best_idx]['alignment_penalty']:.2f}"
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

    evolution_log["final_population"] = [geno.tolist() for geno in population]

    with open(os.path.join(log_dir, "evolution_log.json"), "w") as f:
        json.dump(evolution_log, f, indent=2)

    best_geno = np.array(evolution_log["best_genotype"])
    print(
        f"\nEvolution complete! Best fitness: {max(evolution_log['best_fitness_per_gen']):.2f}\n"
        f"Best genotype: {best_geno}"
    )
    return best_geno, evolution_log