import sys
# sys.path.append(r'G:/STODI/franka-pybullet/src')
import argparse
import os
import os.path as osp
import numpy as np
import copy
import copy
import matplotlib.pyplot as plt
import time
from tqdm import *

from models import RobotArm
from Costs import Multi_Cost
from STO import *
from Visualization import *

def Draw_cost(cost_list):
    x = np.linspace(0, 1, len(cost_list))
    y = cost_list
    plt.plot(x, y, linewidth=1)
    plt.xlabel('Optimized processes (%)')
    plt.ylabel('Cost function value')
    plt.title('Stoil Optimization')
    plt.show()

def write_trajectory(item, path):
    traj_logfile = open(path, 'w')
    for i in range(item.shape[0]):
        traj_logfile.write(" ".join(str(thing) for thing in item[i]))
        traj_logfile.write("\n")
    traj_logfile.close()

def write_joints(item, path):
    traj_logfile = open(path, 'w')
    for i in range(item.shape[0]):
        for j in range(item.shape[1]):
            traj_logfile.write(" ".join(str(thing) for thing in item[i, j]))
            traj_logfile.write(" ")
        traj_logfile.write("\n")
    traj_logfile.close()

def main_single_traj(args):
    # 0. initial the Simulator(Franka, Z1)
    robot = RobotArm(args)

    # 1.需要的输入为: Configuration Space的起点和终点; 使用线性插值的手段生成初始轨迹
    begin_inCspace = [0, 0, 0, -1.6, 0, 1.87, 0]
    end_inCspace = [0, -0.7, 0, -1.6, 0, 3.5, 0.7]
    # begin_inCspace = [-0.3, 0.68, -1.04, 0.18, 0.245, -0.17]
    # end_inCspace = [-0.3, 1, -0.8, 0.18, 0.245, -0.17]
    initial_trajectory = Joint_linear_initial(begin=begin_inCspace, end=end_inCspace)
    '''
    想要模仿的Demonstration
    目前的版本是在Cspace下直接生成轨迹，中间有个中间点进行弯折，比较简单，后期可以留给TODO
    '''
    demostrantion = Generate_demonstration(begin=begin_inCspace, end=end_inCspace)

    # eff = Generate_effector_trajectory(128, 0, 'linear', [0, 0, 0, -1.6, 0, 1.87, 0], [0, -0.7, 0, -1.6, 0, 3.5, 0.7], robot)
    # initial_trajectory = Joint_linear_initial(begin=[0, 0, 0, -1.6, 0, 1.87, 0], end=[0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    # demostrantion = Generate_demonstration_from_effector(eff, robot)
    # print(initial_trajectory.shape, demostrantion.shape)

    init_end_effector = robot.solveListKinematics(initial_trajectory)
    # Cost Function
    cost_function = Multi_Cost(robot=robot, init_trajectory=demostrantion, args=args)
    # Sentry
    sentry = Sentry(initial_trajectory, cost_function, args)
    # 多维Stomp框架，目前这个是Panda的配置
    if args.Robot == 'Panda':
        stomp_py = Multi_dimensions_stomp(
            num_points=initial_trajectory.shape[0], args=args, cost_func=cost_function, 
            rangelimit_low_list=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973]), 
            rangelimit_high_list=np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.752,2.8973]), 
            vellimit_low_list=np.array([-2.1750,-2.1750,-2.1750,-2.1750,-2.6100,-2.6100,-2.6100]), 
            vellimit_high_list=np.array([2.1750,2.1750,2.1750,2.1750,2.6100,2.6100,2.6100]),
            acclimit_low_list=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]), 
            acclimit_high_list=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
            )
    # 导入Z1相关参数，但尚未调通整体框架
    elif args.Robot == 'Z1':
        stomp_py = Multi_dimensions_stomp(
            num_points=initial_trajectory.shape[0], args=args, cost_func=cost_function, 
            rangelimit_low_list=np.array([-2.6179938779914944,0.0,-2.8797932657906435,-1.5184364492350666,-1.3439035240356338,-2.792526803190927]), 
            rangelimit_high_list=np.array([2.6179938779914944,2.9670597283903604,0.0,1.5184364492350666,1.3439035240356338,2.792526803190927]), 
            vellimit_low_list=np.array([-3.1415,-3.1415,-3.1415,-3.1415,-3.1415,-3.1415]), 
            vellimit_high_list=np.array([3.1415,3.1415,3.1415,3.1415,3.1415,3.1415]),
            acclimit_low_list=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]), 
            acclimit_high_list=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]))
    # 远航者的相关配置
    voyager_Qcost_list = []
    voyager_Qcost_total_list = []
    voyager_iter_joints = []

    if args.STO == 'STODI':
        neighbour_Qcost_list = []
        neighbour_Qcost_total_list = []
        neighbour_iter_joints = []

    iter_num = 0
    total_iter_num = args.iter_num

    # 计算Cost标准步骤(for example)
    cost_function.Update_state(generate_multi_state(initial_trajectory, args=args))
    voyager_Qcost_list.append(cost_function.calculate_total_cost(args.ContourCost))
    voyager_Qcost_total_list.append(np.sum(voyager_Qcost_list[-1]))
    voyager_iter_joints.append(initial_trajectory)
    
    if args.STO == 'STODI':
        neighbour_Qcost_list.append(cost_function.calculate_total_cost(args.ContourCost))
        neighbour_Qcost_total_list.append(np.sum(neighbour_Qcost_list[-1]))
        neighbour_iter_joints.append(initial_trajectory)
    
    begin_time = time.time()
    # begin optimal loops
    while(iter_num < total_iter_num):
        iter_num += 1
        # 0. generate noisy trajectory
        stomp_py.multiDimension_diffusion()
        # 1.1 calculate voyager weights
        temp_iter_traj = copy.copy(sentry.pioneer['voyager'].traj)
        stomp_py.multi_update(joints_traj=sentry.pioneer['voyager'].traj)
        kn_weights = stomp_py.calculate_weights()
        n7_proved_noise = stomp_py.calculate_delta_noise(weights_p=kn_weights)

        # 1.2 get new trajectory
        # temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + (args.decay**iter_num) * n7_proved_noise[1:-1, :] # N * 7
        temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + n7_proved_noise[1:-1, :] # N * 7
        # 1.3 limit check
        if stomp_py.multi_limit_check(temp_iter_traj):
            # 1.4 Update Voyager
            sentry.Update('voyager', temp_iter_traj)
            # 1.5 Using for reuse
            if args.reuse_state:
                stomp_py.update_reuse_traj(generate_multi_state(temp_iter_traj, args=args))
            # Cost voyager cost( Just for visualize )
            cost_function.Update_state(generate_multi_state(temp_iter_traj, args=args))
            voyager_Qcost_list.append(cost_function.calculate_total_cost(args.ContourCost))
            voyager_Qcost_total_list.append(np.sum(voyager_Qcost_list[-1]))
            # Record voyager
            voyager_iter_joints.append(copy.copy(temp_iter_traj))
        
        if args.STO == 'STODI':
            # 2.1 calculate neighbour weights
            temp_iter_traj = copy.copy(sentry.pioneer['neighbour'].traj)
            stomp_py.multi_update(joints_traj=sentry.pioneer['neighbour'].traj)
            kn_weights = stomp_py.calculate_weights()
            n7_proved_noise = stomp_py.calculate_delta_noise(weights_p=kn_weights)

            # 2.2 get new trajectory
            temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + (args.decay**iter_num) * n7_proved_noise[1:-1, :] # N * 7
            # temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + n7_proved_noise[1:-1, :] # N * 7
            # 2.3 limit check
            if stomp_py.multi_limit_check(temp_iter_traj):
                # 2.4 Update Neighbour
                sentry.Update('neighbour', temp_iter_traj)
                # 2.5 Using for reuse
                if args.reuse_state:
                    stomp_py.update_reuse_traj(generate_multi_state(temp_iter_traj, args=args))
                # Cost voyager cost( Just for visualize )
                cost_function.Update_state(generate_multi_state(temp_iter_traj, args=args))
                neighbour_Qcost_list.append(cost_function.calculate_total_cost(args.ContourCost))
                neighbour_Qcost_total_list.append(np.sum(neighbour_Qcost_list[-1]))
                # Record voyager
                neighbour_iter_joints.append(copy.copy(temp_iter_traj))
            
            # 3. Update the best
            sentry.Update('best')
            # 4. clock align the best and the neighbour
            if iter_num % (total_iter_num / 10):
                sentry.Update('clock')

    print("Loop is over!")
    Optimization_time = time.time() - begin_time
    print('The total optimization time : {} s.'.format(Optimization_time))
    print(len(voyager_Qcost_list))

    if args.STO == 'STODI':
        final_traj_state = generate_multi_state(sentry.pioneer['best'].traj, args)
    elif args.STO == 'STOMP':
        final_traj_state = generate_multi_state(sentry.pioneer['voyager'].traj, args)
    
    robot.traj_torque_control(final_traj_state["position"], final_traj_state["velocity"], final_traj_state["acceleration"])
    print("演示结束!")
    
    # robot.traj_control(iter_traj)
    # print(Qcost_list)
    # Draw_cost(voyager_Qcost_total_list)
    # Draw_cost(neighbour_Qcost_total_list)
    # Draw_cost(list(np.array(neighbour_Qcost_list)[:,1]))

    if args.STO == 'STODI':
        end_effector = robot.solveListKinematics(sentry.pioneer['best'].traj)
    elif args.STO == 'STOMP':
        end_effector = robot.solveListKinematics(sentry.pioneer['voyager'].traj)
    
    eff = np.array(robot.solveListKinematics(demostrantion))
    Draw_3trajectory(init_end_effector, end_effector, eff)
    # # print(end_effector.shape)
    # write_trajectory(end_effector[:, :3], f"{args.file_path}/results/{args.expt_name}/trajectory_logs.txt")
    # write_joints(np.array(neighbour_iter_joints), f"{args.file_path}/results/{args.expt_name}/joints_logs.txt")

    # logfile = open(f"{args.file_path}/results/{args.expt_name}/result_logs.txt", 'w')
    # logfile.write("\n".join(str(item) for item in neighbour_Qcost_list))
    # logfile.close()

def stochastic_optimization_stomp(compare, stomp_py, sentry, cost_function, record_list, args):
    voyager_Qcost_list = copy.copy(record_list[0])
    voyager_iter_joints = copy.copy(record_list[1])
    # 以进行优化的轮次作为结束的标志
    if compare == args.iter_num:
        iter_num = 0
        total_iter_num = args.iter_num
        begin_time = time.time()
        # begin optimal loops
        while(iter_num < total_iter_num):
            iter_num += 1
            # 0. generate noisy trajectory
            stomp_py.multiDimension_diffusion()
            # 1.1 calculate voyager weights
            temp_iter_traj = copy.copy(sentry.pioneer['voyager'].traj)
            stomp_py.multi_update(joints_traj=sentry.pioneer['voyager'].traj)
            kn_weights = stomp_py.calculate_weights()
            n7_proved_noise = stomp_py.calculate_delta_noise(weights_p=kn_weights)

            # 1.2 get new trajectory
            # temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + (args.decay**iter_num) * n7_proved_noise[1:-1, :] # N * 7
            temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + n7_proved_noise[1:-1, :] # N * 7
            # 1.3 limit check
            if stomp_py.multi_limit_check(temp_iter_traj):
                # 1.4 Update Voyager
                sentry.Update('voyager', temp_iter_traj)
                # 1.5 Using for reuse
                if args.reuse_state:
                    stomp_py.update_reuse_traj(generate_multi_state(temp_iter_traj, args=args))
                # Cost voyager cost( Just for visualize )
                cost_function.Update_state(generate_multi_state(temp_iter_traj, args=args))
                voyager_Qcost_list.append(cost_function.calculate_total_cost(args.ContourCost))
                # Record voyager
                voyager_iter_joints.append(copy.copy(temp_iter_traj))
        Optimization_time = time.time() - begin_time
        return Optimization_time, voyager_Qcost_list, voyager_iter_joints
    # 以固定时间作为结束的标志
    else:
        iter_num = 0
        begin_time = time.time()
        while(1):
            iter_num += 1
            stomp_py.multiDimension_diffusion(iter_num)
            # 1.1 calculate voyager weights
            temp_iter_traj = copy.copy(sentry.pioneer['voyager'].traj)
            stomp_py.multi_update(joints_traj=sentry.pioneer['voyager'].traj)
            kn_weights = stomp_py.calculate_weights()
            n7_proved_noise = stomp_py.calculate_delta_noise(weights_p=kn_weights)

            # 1.2 get new trajectory
            # temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + (args.decay**iter_num) * n7_proved_noise[1:-1, :] # N * 7
            temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + n7_proved_noise[1:-1, :] # N * 7
            # 1.3 limit check
            if stomp_py.multi_limit_check(temp_iter_traj):
                # 1.4 Update Voyager
                sentry.Update('voyager', temp_iter_traj)
                # 1.5 Using for reuse
                if args.reuse_state:
                    stomp_py.update_reuse_traj(generate_multi_state(temp_iter_traj, args=args))
                # Cost voyager cost( Just for visualize )
                cost_function.Update_state(generate_multi_state(temp_iter_traj, args=args))
                voyager_Qcost_list.append(cost_function.calculate_total_cost(args.ContourCost))
                # Record voyager
                voyager_iter_joints.append(copy.copy(temp_iter_traj))
            if time.time() - begin_time > compare:
                break
        Optimization_time = time.time() - begin_time
        return Optimization_time, voyager_Qcost_list, voyager_iter_joints

def stochastic_optimization_stodi(stomp_py, sentry, cost_function, record_list, args):
    neighbour_Qcost_list = copy.copy(record_list[0])
    neighbour_iter_joints = copy.copy(record_list[1])
    iter_num = 0
    total_iter_num = args.iter_num
    begin_time = time.time()
    # begin optimal loops
    while(iter_num < total_iter_num):
        iter_num += 1
        # 0. generate noisy trajectory
        stomp_py.multiDimension_diffusion(iter_num)
        # 1.1 calculate voyager weights
        temp_iter_traj = copy.copy(sentry.pioneer['voyager'].traj)
        stomp_py.multi_update(joints_traj=sentry.pioneer['voyager'].traj)
        kn_weights = stomp_py.calculate_weights()
        n7_proved_noise = stomp_py.calculate_delta_noise(weights_p=kn_weights)

        # 1.2 get new trajectory
        # temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + (args.decay**iter_num) * n7_proved_noise[1:-1, :] # N * 7
        temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + n7_proved_noise[1:-1, :] # N * 7
        # 1.3 limit check
        if stomp_py.multi_limit_check(temp_iter_traj):
            # 1.4 Update Voyager
            sentry.Update('voyager', temp_iter_traj)
            # 1.5 Using for reuse
            if args.reuse_state:
                stomp_py.update_reuse_traj(generate_multi_state(temp_iter_traj, args=args))

        # 2.1 calculate neighbour weights
        temp_iter_traj = copy.copy(sentry.pioneer['neighbour'].traj)
        stomp_py.multi_update(joints_traj=sentry.pioneer['neighbour'].traj)
        kn_weights = stomp_py.calculate_weights()
        n7_proved_noise = stomp_py.calculate_delta_noise(weights_p=kn_weights)

        # 2.2 get new trajectory
        temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + (args.decay**iter_num) * n7_proved_noise[1:-1, :] # N * 7
        # temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + n7_proved_noise[1:-1, :] # N * 7
        # 2.3 limit check
        if stomp_py.multi_limit_check(temp_iter_traj):
            # 2.4 Update Neighbour
            sentry.Update('neighbour', temp_iter_traj)
            # 2.5 Using for reuse
            if args.reuse_state:
                stomp_py.update_reuse_traj(generate_multi_state(temp_iter_traj, args=args))
            # Cost voyager cost( Just for visualize )
            cost_function.Update_state(generate_multi_state(temp_iter_traj, args=args))
            neighbour_Qcost_list.append(cost_function.calculate_total_cost(args.ContourCost))
            # Record voyager
            neighbour_iter_joints.append(copy.copy(temp_iter_traj))
        # 3. Update the best
        sentry.Update('best')
        # 4. clock align the best and the neighbour
        if iter_num % (total_iter_num / 10):
            sentry.Update('clock')
    Optimization_time = time.time() - begin_time

    return Optimization_time, neighbour_Qcost_list, neighbour_iter_joints

def main(times, args):
    # 0. initial the Simulator(Franka, Z1)
    robot = RobotArm(args)

    # 1.需要的输入为: Configuration Space的起点和终点; 使用线性插值的手段生成初始轨迹
    begin_inCspace = [0, 0, 0, -1.6, 0, 1.87, 0]
    end_inCspace = [0, -0.7, 0, -1.6, 0, 3.5, 0.7]
    # begin_inCspace = [-0.3, 0.68, -1.04, 0.18, 0.245, -0.17]
    # end_inCspace = [-0.3, 1, -0.8, 0.18, 0.245, -0.17]
    initial_trajectory = Joint_linear_initial(begin=begin_inCspace, end=end_inCspace)
    '''
    想要模仿的Demonstration
    目前的版本是在Cspace下直接生成轨迹，中间有个中间点进行弯折，比较简单，后期可以留给TODO
    '''
    demostrantion = Generate_demonstration(begin=begin_inCspace, end=end_inCspace)

    init_end_effector = robot.solveListKinematics(initial_trajectory)
    # Cost Function
    cost_function = Multi_Cost(robot=robot, init_trajectory=demostrantion, args=args)
    # Sentry
    sentry = Sentry(initial_trajectory, cost_function, args)
    # 多维Stomp框架，目前这个是Panda的配置
    if args.Robot == 'Panda':
        stomp_py = Multi_dimensions_stomp(
            num_points=initial_trajectory.shape[0], args=args, cost_func=cost_function, 
            rangelimit_low_list=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973]), 
            rangelimit_high_list=np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.752,2.8973]), 
            vellimit_low_list=np.array([-2.1750,-2.1750,-2.1750,-2.1750,-2.6100,-2.6100,-2.6100]), 
            vellimit_high_list=np.array([2.1750,2.1750,2.1750,2.1750,2.6100,2.6100,2.6100]),
            acclimit_low_list=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]), 
            acclimit_high_list=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
            )
    # 导入Z1相关参数，但尚未调通整体框架
    elif args.Robot == 'Z1':
        stomp_py = Multi_dimensions_stomp(
            num_points=initial_trajectory.shape[0], args=args, cost_func=cost_function, 
            rangelimit_low_list=np.array([-2.6179938779914944,0.0,-2.8797932657906435,-1.5184364492350666,-1.3439035240356338,-2.792526803190927]), 
            rangelimit_high_list=np.array([2.6179938779914944,2.9670597283903604,0.0,1.5184364492350666,1.3439035240356338,2.792526803190927]), 
            vellimit_low_list=np.array([-3.1415,-3.1415,-3.1415,-3.1415,-3.1415,-3.1415]), 
            vellimit_high_list=np.array([3.1415,3.1415,3.1415,3.1415,3.1415,3.1415]),
            acclimit_low_list=np.array([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]), 
            acclimit_high_list=np.array([np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]))
    # 远航者的多次实验的完整实验记录
    voyager_Qcost_TL = []
    voyager_joints_TL = []

    neighbour_Qcost_TL = []
    neighbour_joints_TL = []

    Qcost_list = []
    iter_joints = []
    # 计算Cost标准步骤(for example)
    cost_function.Update_state(generate_multi_state(initial_trajectory, args=args))
    Qcost_list.append(cost_function.calculate_total_cost(args.ContourCost))
    iter_joints.append(initial_trajectory)
    
    record_list = [Qcost_list, iter_joints]

    avg_time = 0
    for i in tqdm(range(times)):
        time, Q, Traj = stochastic_optimization_stodi(stomp_py, sentry, cost_function, record_list, args)
        avg_time = avg_time + time
        neighbour_Qcost_TL.append(Q)
        neighbour_joints_TL.append(Traj)

        # 重置初始条件
        sentry.New_turn(initial_trajectory)
        stomp_py.Reset()
    
    avg_time = avg_time / times
    print(avg_time)

    for i in tqdm(range(times)):
        time, Q, Traj = stochastic_optimization_stomp(avg_time, stomp_py, sentry, cost_function, record_list, args)
        voyager_Qcost_TL.append(Q)
        voyager_joints_TL.append(Traj)

        # 重置初始条件
        sentry.New_turn(initial_trajectory)
        stomp_py.Reset()
    
    Draw_Cost(voyager_Qcost_TL, neighbour_Qcost_TL, avg_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--file-path", type=str, default=r'./Pybullet')
    parser.add_argument("--expt-name", type=str, required=True)
    # STO参数
    parser.add_argument("--Robot", type=str, choices=["Panda", "Z1"], default="Panda") # 选择机器人模型
    parser.add_argument("--dimension", type=int, default=7) # 输入的维数:tips Panda是7，Z1是6
    parser.add_argument("--iter-num", type=int, default=20) # STO迭代的次数
    parser.add_argument("--reuse-state", type=bool, default=True) # 是否启用reuse
    parser.add_argument("--K", type=int, default=20) # STO选取的K条轨迹的数量 20
    parser.add_argument("--reuse-num", type=int, default=10) # reuse的数量 10
    parser.add_argument("--decay", type=float, default=0.9) # decay for better 收敛 0.96
    parser.add_argument("--STO", type=str, choices=["STODI", "STOMP"], default="STODI") # 是什么随机优化框架
    # loss参数
    parser.add_argument("--ContourCost", type=str, choices=["NONE", "DTW", "MSES", "MSEPS", "NMSEPS", "MSE"], default="DTW") # 模仿学习的loss函数指标选择
    parser.add_argument("--ObstacleCost", type=str, choices=["NONE", "STOMP"], default="STOMP") # 避障的loss
    parser.add_argument("--ConstraintCost", type=str, choices=["NONE", "STOMP"], default="STOMP") # 约束的loss
    parser.add_argument("--TorqueCost", type=str, choices=["NONE", "STOMP"], default="STOMP") # 力矩的loss
    # 结果可视化参数
    parser.add_argument("--visual-loss", action="store_true")
    parser.add_argument("--visual-traj", action="store_true")
    # 其他参数
    parser.add_argument("--control-frequency", type=float, default=30) # 假定的控制频率
    parser.add_argument("--sample-frequency", type=float, default=30) # 假定的采样频率

    args = parser.parse_args()

    if not osp.isdir(f"{args.file_path}/results/"):
        os.makedirs(f"{args.file_path}/results/")

    if not osp.isdir(f"{args.file_path}/results/{args.expt_name}/"):
        os.makedirs(f"{args.file_path}/results/{args.expt_name}/")
    # 单次选择运行（STOMP,STODI）的主函数
    # main_single_traj(args=args)
    main(10, args)