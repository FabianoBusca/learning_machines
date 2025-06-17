from pathlib import Path

import cv2
import random
import json
from datetime import datetime

from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

FIGURES_DIR = Path("/root/results")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def test_emotions(rob: IRobobo):
    """Tests emotion setting, talking, sound playback, and LED color change."""
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)

def test_move_and_wheel_reset(rob: IRobobo):
    """Tests robot movement and wheel reset functionality."""
    rob.move_blocking(100, 100, 1000)
    print("Before reset:", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("After reset:", rob.read_wheels())

def test_sensors(rob: IRobobo):
    """Reads and logs all sensor data and saves a front image."""
    print("IRS data:", rob.read_irs())
    image = rob.read_image_front()
    cv2.imwrite(str(FIGURES_DIR / "photo.png"), image)
    print("Phone pan:", rob.read_phone_pan())
    print("Phone tilt:", rob.read_phone_tilt())
    print("Current acceleration:", rob.read_accel())
    print("Current orientation:", rob.read_orientation())

def test_phone_movement(rob: IRobobo):
    """Tests phone pan and tilt commands."""
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20:", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50:", rob.read_phone_tilt())

def test_sim(rob: SimulationRobobo):
    """Tests simulation control and state reporting."""
    print("Current simulation time:", rob.get_sim_time())
    print("Simulation running:", rob.is_running())
    rob.stop_simulation()
    print("Simulation time after stop:", rob.get_sim_time())
    print("Running after stop:", rob.is_running())
    rob.play_simulation()
    print("Simulation time after restart:", rob.get_sim_time())
    print("Robot position:", rob.get_position())
    print("Robot orientation:", rob.get_orientation())

    pos = rob.get_position()
    orient = rob.get_orientation()
    rob.set_position(pos, orient)
    print("Position unchanged:", pos == rob.get_position())
    print("Orientation unchanged:", orient == rob.get_orientation())

def test_hardware(rob: HardwareRobobo):
    """Checks hardware battery levels."""
    print("Phone battery:", rob.phone_battery())
    print("Robot battery:", rob.robot_battery())

def run_all_actions(rob: IRobobo):
    """
    Runs a test loop that checks proximity with IR sensors,
    triggers obstacle avoidance behavior, and logs activation values.
    """
    if isinstance(rob, SimulationRobobo):
        try:
            if not rob.is_running():
                print("Starting simulation...")
                rob.play_simulation()
                rob.sleep(2)
            else:
                print("Simulation already running.")
        except Exception as e:
            print(f"Simulation init error: {e}")
            return

    activation = []
    raw_values = []
    all_activation = []
    all_raw_sensor_data = []
    touch = False
    turn_direction = "left"

    # for i in range(5):
    #     rob.move_blocking(100, 100, 1000)
    #
    # if isinstance(rob, SimulationRobobo):
    #     try:
    #         rob.stop_simulation()
    #         print("Simulation stopped.")
    #     except Exception as e:
    #         print(f"Error stopping simulation: {e}")

    for i in range(15):
        try:
            irs = rob.read_irs()
            if irs is None or len(irs) < 5:
                print(f"[{i}] Invalid IR sensor data")
                rob.sleep(0.5)
                continue

            front_center = irs[4]
            print(f"[{i}] Front IR reading: {front_center}")

            act = [1 if val >= 250 else 0 for val in irs]
            activation.append(act)
            raw_values.append(irs)

            if front_center >= 250:
                rob.talk("Obstacle ahead")
                print("Too close! Reversing...")
                rob.move_blocking(-128, -128, 1000)
                touch = True
                rob.sleep(0.5)
                turn_direction = random.choice(["left", "right"])
                if turn_direction == "left":
                    print("Turning left...")
                    rob.move_blocking(-50, 100, 800)
                else:
                    print("Turning right...")
                    rob.move_blocking(100, -50, 800)
            else:
                if touch and turn_direction == "left":
                    rob.move_blocking(-50, 100, 800)
                    rob.talk("Turning left")
                elif touch and turn_direction == "right":
                    rob.move_blocking(100, -50, 800)
                    rob.talk("Turning right")
                else:
                    rob.move_blocking(100, 100, 1000)

        except Exception as e:
            print(f"Error at step {i}: {e}")
            rob.sleep(1)
            continue

    all_activation.append(activation)
    all_raw_sensor_data.append(raw_values)

    print("Activation log:", activation)
    print("Raw IR values:", raw_values)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = FIGURES_DIR / f"raw_ir_data_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump({"raw_ir_data": raw_values}, f, indent=2)
    print(f"Raw sensor data saved to {out_file}")

    if isinstance(rob, SimulationRobobo):
        try:
            rob.stop_simulation()
            print("Simulation stopped.")
        except Exception as e:
            print(f"Error stopping simulation: {e}")