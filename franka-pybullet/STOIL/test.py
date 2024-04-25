import sys
sys.path.append(r'D:/STOIL/src/franka-pybullet/src')
import argparse
# import pybullet as p
import time
from math import sin
from generate_initial_traj import Joint_linear_initial, Generate_demonstration
from main import generate_multi_state, Draw_cost
from robot_model import Panda
from contour_cost import FFT
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default=r'D:/STOIL/src/franka-pybullet/src')

    parser.add_argument("--dimension", type=int, default=7)
    parser.add_argument("--add-contourCost", type=bool, default=False)
    parser.add_argument("--reuse-state", type=bool, default=True)
    parser.add_argument("--reuse-num", type=int, default=5)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--decay", type=float, default=0.99)

    parser.add_argument("--control-frequency", type=float, default=10)
    parser.add_argument("--sample-frequency", type=float, default=20)
    
    args = parser.parse_args()


    duration = 1
    stepsize = 1e-3

    robot = Panda(stepsize)
    initial = Joint_linear_initial(begin=[0, 0, 0, -1.6, 0, 1.87, 0], end=[0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    initial_state = generate_multi_state(initial, args)

    demostration = Generate_demonstration(begin=[0, 0, 0, -1.6, 0, 1.87, 0], end=[0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    demostration_state = generate_multi_state(demostration, args)

    robot.traj_torque_control(initial_state["position"], initial_state["velocity"], initial_state["acceleration"])
    print("motion success!")
    robot.reset()
    robot.traj_torque_control(demostration_state["position"], demostration_state["velocity"], demostration_state["acceleration"])