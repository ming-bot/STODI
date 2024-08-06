import sys
sys.path.append(r'E:/STODI/src/franka-pybullet/src')
import argparse
# import pybullet as p
import time
from math import sin
from generate_initial_traj import Joint_linear_initial, Generate_demonstration
from main import generate_multi_state, Draw_cost
from visualization import Draw_3trajectory
from robot_model import Panda
from contour_cost import FFT
import numpy as np

def read_state(path):
    state_logfile = open(path, 'r')
    content = state_logfile.readlines()
    state_logfile.close()
    return content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default=r'E:/STODI/src/franka-pybullet/src')

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
    # initial = Joint_linear_initial(begin=[0, 0, 0, -1.6, 0, 1.87, 0], end=[0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    # initial_state = generate_multi_state(initial, args)

    # demostration = Generate_demonstration(begin=[0, 0, 0, -1.6, 0, 1.87, 0], end=[0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    # demostration_state = generate_multi_state(demostration, args)

    # robot.traj_torque_control(initial_state["position"], initial_state["velocity"], initial_state["acceleration"])
    # print("motion success!")
    # robot.reset()
    # robot.traj_torque_control(demostration_state["position"], demostration_state["velocity"], demostration_state["acceleration"])

    path = r'E:/STODI/franka-pybullet/src/results/0606stomp/trajectory_logs.txt'
    traj = read_state(path)
    for i in range(len(traj)):
        traj[i] = traj[i].split()
    traj = np.array(traj).astype(np.float64)
    
    demostrantion = Generate_demonstration(begin=[0, 0, 0, -1.6, 0, 1.87, 0], end=[0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    demostrantion_end_effector = np.array(robot.solveListKinematics(demostrantion))


    initial_trajectory = Joint_linear_initial(begin=[0, 0, 0, -1.6, 0, 1.87, 0], end=[0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    init_end_effector = robot.solveListKinematics(initial_trajectory)

    Draw_3trajectory(init_end_effector, traj, demostrantion_end_effector)