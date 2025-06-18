#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions, race, evolve_population, explore, collect, evolve_eater_population#, learn

if __name__ == "__main__":
    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    # run_all_actions(rob)
    # race(rob, steps=150)
    # evolve_population(rob, steps_per_race=60, pop_size=20, generations=100)
    # explore(rob, cells=50)
    # learn(rob)
    collect(rob)
    # evolve_eater_population(rob, generations=100)