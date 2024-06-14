# 依赖于 robot_model.py
# 记录机器人的轨迹数据
import sys
sys.path.append(r'E:/Proud/franka-pybullet/src')

from robot_model import Panda
import time
import json  # 用于保存记录的数据

t = 0.0
stepsize = 1e-3
realtime = 0

robot = Panda()
robot.demo_mode()

# 初始化记录结构
trajectory_data = []

# 仿真循环
try:
    while True:
        # 获取所有关节的状态
        joint_states = robot.getJointStates()

        # 提取所有关节的位置和速度
        joint_positions = joint_states[0]
        joint_velocities = joint_states[1]

        # 记录当前时间步的关节状态
        trajectory_data.append({
            'time': time.time(),
            'positions': joint_positions,
            'velocities': joint_velocities
        })

        # 数据采样间隔 0.01 秒
        time.sleep(0.01)

except KeyboardInterrupt:
    # 保存记录的数据到文件
    with open(r'E:/Proud/franka-pybullet/Demo/trajectories/aaa.json', 'w') as f:
        json.dump(trajectory_data, f, indent=4)
    print("Trajectory data has been saved to 'aaa.json'")

finally:
    # 断开连接
    robot.disconnect()







