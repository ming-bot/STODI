from robot_model import Panda
# import sys
# sys.path.append('../src')
# from panda import Panda
import time
import numpy as np
import pybullet as p
from math import sin


def read_state(path):
    state_logfile = open(path, 'r')
    content = state_logfile.read()
    state_logfile.close()
    return content


if __name__ == "__main__":
    robot = Panda()
    robot.setControlMode("torque")

    # 读取规划好的轨迹状态，获得 array 形式的 pos_desired, vel_desired, acc_desired
    path = 'E:/Proud/franka-pybullet/src/results/0424'
    pos_str, vel_str, acc_str = read_state(path + "/position.txt"), read_state(path + "/velocity.txt"), read_state(
        path + "/acceleration.txt")  # str 格式
    pos_tmp, vel_tmp, acc_tmp = np.array(pos_str.split()), np.array(vel_str.split()), np.array(
        acc_str.split())  # array 格式
    pos_desired, vel_desired, acc_desired = pos_tmp.astype(np.float64).reshape((256, 7)), vel_tmp.astype(
        np.float64).reshape((256, 7)), acc_tmp.astype(np.float64).reshape((256, 7))
    initial_pos = pos_desired[0]

    # 开始录制 pybullet 视频
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, path + "/robotmove.mp4")

    robot.traj_torque_control(pos_desired, vel_desired, acc_desired)
    # 结束录制 pybullet 视频
    p.stopStateLogging(log_id)

