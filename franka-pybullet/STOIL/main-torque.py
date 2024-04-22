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
    path = 'C:/Users/hp/PycharmProjects/trajOptimize/franka-pybullet/src/results/0419new'
    pos_str, vel_str, acc_str = read_state(path + "/position.txt"), read_state(path + "/velocity.txt"), read_state(
        path + "/acceleration.txt")  # str 格式
    pos_tmp, vel_tmp, acc_tmp = np.array(pos_str.split()), np.array(vel_str.split()), np.array(
        acc_str.split())  # array 格式
    pos_desired, vel_desired, acc_desired = pos_tmp.astype(np.float64).reshape((256, 7)), vel_tmp.astype(
        np.float64).reshape((256, 7)), acc_tmp.astype(np.float64).reshape((256, 7))
    initial_pos = pos_desired[0]

    # 开始录制 pybullet 视频
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, path + "/robotmove.mp4")

    # torque control 的代码先不写在 traj_torque_control 里
    # robot.traj_torque_control(robot, pos_desired, vel_desired, acc_desired)
    duration = 4  # 256*0.025
    num = 0
    stepsize = 1e-3

    for i in range(int(duration / stepsize)):
        if i % 100 == 0:
            print("Simulation time: {:.3f}".format(robot.t))

        if i % 10000 == 0:
            robot.reset()
            robot.setControlMode("torque")

        # if i % 5 == 0 and num <= 255:
        if num <= 255:
            pos_desired_update = pos_desired[num]
            vel_desired_update = vel_desired[num]
            acc_desired_update = acc_desired[num]
            num += 1

        # 获取仿真中机械臂关节的 position 和 velocity
        pos_simulated, vel_simulated = robot.getJointStates()

        # 设计预期 acceleration
        acc = [0 for x in pos_simulated]
        kv, kp = 100, 100   # 位置和速度的比例控制器增益矩阵，但我懒就先写 1 假装是单位矩阵好了
        acc_feedback = list(10*(acc_desired_update - kv * (np.array(vel_simulated) - vel_desired_update) - kp * (
                np.array(pos_simulated) - pos_desired_update)))
        # print(acc_feedback)
        # print(acc_desired_update)

        # 计算并应用 torque
        # solveInverseDynamics 的输入必须是 list 格式的
        joints_torque = robot.solveInverseDynamics(pos_simulated, vel_simulated, acc_feedback)  #acc
        # joints_torque = [100, 100, 100, 0, 100, 100, 0]
        # print(joints_torque)
        robot.setTargetTorques(joints_torque)

        robot.step()
        time.sleep(robot.stepsize)

    # 结束录制 pybullet 视频
    p.stopStateLogging(log_id)

