# 依赖于 robot_model.py
# 记录机器人的轨迹数据
from models import RobotArm
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径参数
    # parser.add_argument("--file-path", type=str, default=r'./Pybullet')
    # parser.add_argument("--expt-name", type=str, required=True)
    # STO参数
    parser.add_argument("--Robot", type=str, choices=["Panda", "Z1"], default="Panda") # 选择机器人模型
    parser.add_argument("--dimension", type=int, default=7) # 输入的维数:tips Panda是7，Z1是6
    parser.add_argument("--iter-num", type=int, default=20) # STO迭代的次数
    parser.add_argument("--reuse-state", type=bool, default=True) # 是否启用reuse
    parser.add_argument("--K", type=int, default=20) # STO选取的K条轨迹的数量 20
    parser.add_argument("--reuse-num", type=int, default=10) # reuse的数量 10
    parser.add_argument("--time", type=float, default=0.0) # STO的迭代时间
    parser.add_argument("--decay", type=float, default=0.6) # decay for better收敛
    parser.add_argument("--STO", type=str, choices=["STODI", "STOMP"], default="STODI") # 是什么随机优化框架
    # loss参数
    parser.add_argument("--ContourCost", type=str, choices=[None, "DTW", "MSES", "MSEPS", "NMSEPS", "MSE"], default="MSES") # 模仿学习的loss函数指标选择
    parser.add_argument("--ObstacleCost", type=str, choices=[None, "STOMP"], default="STOMP") # 避障的loss
    parser.add_argument("--ConstraintCost", type=str, choices=[None, "STOMP"], default="STOMP") # 约束的loss
    parser.add_argument("--TorqueCost", type=str, choices=[None, "STOMP"], default="STOMP") # 力矩的loss
    # 结果可视化参数
    parser.add_argument("--visual-loss", action="store_true")
    parser.add_argument("--visual-traj", action="store_true")
    # 其他参数
    parser.add_argument("--control-frequency", type=float, default=30) # 假定的控制频率
    parser.add_argument("--sample-frequency", type=float, default=30) # 假定的采样频率

    args = parser.parse_args()

    t = 0.0
    stepsize = 1e-3
    realtime = 0

    robot = RobotArm(args)
    robot.demo_mode()


    # 仿真循环
    try:
        while True:
            # 获取所有关节的状态
            joint_states = robot.getJointStates()

            # 提取所有关节的位置和速度
            joint_positions = joint_states[0]
            # joint_velocities = joint_states[1]

            # 记录当前时间步的关节状态
            print('positions: ', joint_positions)

            # 数据采样间隔 0.01 秒
            time.sleep(0.01)

    except KeyboardInterrupt:
        # 保存记录的数据到文件
        # with open(r'C:/Users/hp/PycharmProjects/trajOptimize/franka-pybullet/src/results/0530/aaa.json', 'w') as f:
        #     json.dump(trajectory_data, f, indent=4)
        print("Trajectory data has been saved to 'aaa.json'")

    finally:
        # 断开连接
        robot.disconnet()







