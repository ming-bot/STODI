import sys
sys.path.append(r'D:/STOIL/src/franka-pybullet/src')
import argparse
# import pybullet as p
import time
from math import sin
from generate_initial_traj import Joint_linear_initial
from main import generate_multi_state, Draw_cost
# from robot_model import Panda
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

    signal = np.random.rand(256, 3)
    print(signal.shape)
    fft_signal = FFT(signal)
    print(fft_signal.shape)
    # Draw_cost(fft_signal[2, :])
    time_energ = np.sum(signal * signal)
    freq_energ = (1.0 / (256 * 3)) * np.sum(fft_signal * fft_signal)

    print(time_energ, freq_energ)
    print(time_energ - freq_energ)

    # duration = 1
    # stepsize = 1e-3

    # robot = Panda(stepsize)
    # robot.setControlMode("velocity")
    # initial = Joint_linear_initial(begin=[0, 0, 0, -1.6, 0, 1.87, 0], end=[0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    # car_traj = robot.solveListKinematics(initial)
    # state_traj = generate_multi_state(car_traj, args)
    # print(state_traj)
    # init_state = generate_multi_state(initial, args)
    # num = 0

    # for i in range(int(duration/stepsize)):
    #     # robot.setTargetPositions([0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    #     # if i%1000 == 0:
    #     #     print("Simulation time: {:.3f}".format(robot.t))
    #     if i % int(0.025 / stepsize) and i < 256:
    #         robot.setTargetVelocity(init_state["velocity"][num, :])
    #         num += 1
    #         # print(robot.getJointStates())
    #         # target_pos = robot.solveInverseKinematics(ball_pos,[1,0,0,0])
    #         # robot.setTargetPositions(target_pos)
    #         # print(robot.getJointStates())
    #     robot.step()
    #     time.sleep(robot.stepsize)

    #     print(robot.getJointStates())