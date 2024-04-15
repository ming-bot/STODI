from robot_model import Panda
import numpy as np
import pybullet as p
import faulthandler


def read_state(path):
    state_logfile = open(path, 'r')
    content = state_logfile.read()
    state_logfile.close()
    return content


if __name__ == "__main__":
    faulthandler.enable()  # start @ the beginning
    robot = Panda()
    robot.setControlMode("position")

    path = 'C:/Users/hp/PycharmProjects/trajOptimize/franka-pybullet/src/results/0415'
    pos_str, vel_str, acc_str = read_state(path+"/position.txt"), read_state(path+"/velocity.txt"), read_state(path+"/acceleration.txt")      # str 格式
    pos_tmp, vel_tmp, acc_tmp = np.array(pos_str.split()), np.array(vel_str.split()), np.array(acc_str.split())    # list 格式
    position, velocity, acceleration = pos_tmp.astype(np.float64).reshape((256, 7)), vel_tmp.astype(np.float64).reshape((256, 7)), acc_tmp.astype(np.float64).reshape((256, 7))
    print(acceleration.shape)

    torques = robot.solveInverseDynamics(position[0], velocity[0], acceleration[1])
    # torques = robot.solveInverseDynamics(pos=position, vel=velocity, acc=acceleration)
    # print(torques)
    # robot.traj_torque_control(torques)
