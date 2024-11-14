import argparse
import os
import os.path as osp
import numpy as np
import copy
import time
import json
from tqdm import *

from models import RobotArm
from Costs import Multi_Cost
from STO import *
from Visualization import *

POSTURE = {'Panda': {'front': [-0.4873441065691155, 0.5814711977864513, 0.9234217980183635, -1.2212066335195748, -0.3227819795704253, 2.247907005921566, 0.18368041341779082],
           'back':[-0.43700637536473935, -1.4892142186045723, 0.4836279074937232, -2.19738087759112, 0.8110974564673381, 0.6360887472170061, 0.3197152547171901],
           'left': [0.3884776329219501, -0.9546968008083578, -0.9810819639199435, -1.9271960854579653, -0.8515225212999344, 1.6225280247076888, -0.4879094307978122],
           'middle':[0, 0, 0, -1.6, 0, 1.87, 0],
           'up': [0.0, -0.7, 0.0, -1.6, 0.0, 3.5, 0.7],
           'right':[-0.3948859155257995, -0.7270775750246515, 0.9705156650028227, -1.7756625474152135, -0.2109975244307001, 2.7461024636268965, 1.071975144388428],
           },
           'Z1':{'front': [-0.4801968051411831, 2.4705191500590398, -2.2298841544695502, -0.4603309931412034, -0.26550765582356034, -1.7948440075195375],
           'back':[-0.16095450339344913, 2.03714477446206e-05, -0.1206929181796797, 0.23051021514256406, 0.12173945741366415, -0.29745731880453813],
           'left': [-1.0224235397243595, 1.57891271955836, -0.9244010364725801, -0.39626622742773004, -0.3769407847833072, 0.26362689271279327],
           'middle':[-0.30084089082422866, 0.7460330682966394, -0.3125014581909603, -0.3117413872715204, 0.3659345169069073, -1.2592826882667836],
           'up': [-0.3, 0.68, -1.04, 0.18, 0.245, -0.17],
           'right':[0.9047114836368709, 1.6134905356772113, -1.1030674545303771, -0.08813445406564796, 0.20095661087100894, 0.6203964876740342],
           }
           }

def demo(args):
    # 0. initial the Simulator(Franka, Z1)
    robot = RobotArm(args)

    # 1.需要的输入为: Configuration Space的起点和终点; 使用线性插值的手段生成初始轨迹
    begin_inCspace = POSTURE[args.Robot]['up']
    end_inCspace = POSTURE[args.Robot]['front']
    initial_trajectory = Joint_linear_initial(begin=begin_inCspace, end=end_inCspace)
    '''
    想要模仿的Demonstration
    目前的版本是在Cspace下直接生成轨迹，中间有个中间点进行弯折，比较简单，后期可以留给TODO
    '''
    external = POSTURE[args.Robot]['right']
    demostrantion = Generate_demo_demonstration(begin=begin_inCspace, end=end_inCspace, external=external, dimension=args.dimension)

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
        temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + (args.decay**iter_num) * n7_proved_noise[1:-1, :] # N * 7
        # temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + n7_proved_noise[1:-1, :] # N * 7
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
            temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + (args.decay**iter_num) * 0.5 * n7_proved_noise[1:-1, :] # N * 7
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

    if args.STO == 'STODI':
        end_effector = robot.solveListKinematics(sentry.pioneer['best'].traj)
    elif args.STO == 'STOMP':
        end_effector = robot.solveListKinematics(sentry.pioneer['voyager'].traj)
    
    demonstration_eff = robot.solveListKinematics(demostrantion)
    Draw_3trajectory(init_end_effector, end_effector, demonstration_eff)

    writing_results = {
        'init_traj': init_end_effector.tolist(),
        'result_traj': end_effector.tolist(),
        'demonstration': demonstration_eff.tolist(),
    }
    
    if args.STO == 'STODI':
        writing_results['cost'] = np.array(neighbour_Qcost_list).tolist()
    elif args.STO == 'STOMP':
        writing_results['cost'] = np.array(voyager_Qcost_list).tolist()
    
    with open(f"{args.file_path}/demos/{args.expt_name}.json", 'w') as f:
        json.dump(writing_results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--file-path", type=str, default=r'./Pybullet')
    parser.add_argument("--expt-name", type=str, required=True)
    # STO参数
    parser.add_argument("--Robot", type=str, choices=["Panda", "Z1"], default="Z1") # 选择机器人模型
    parser.add_argument("--dimension", type=int, default=6) # 输入的维数:tips Panda是7，Z1是6
    parser.add_argument("--iter-num", type=int, default=10) # STO迭代的次数
    parser.add_argument("--reuse-state", type=bool, default=True) # 是否启用reuse
    parser.add_argument("--K", type=int, default=20) # STO选取的K条轨迹的数量 20
    parser.add_argument("--reuse-num", type=int, default=10) # reuse的数量 10
    parser.add_argument("--time", type=float, default=0.0) # STO的迭代时间
    parser.add_argument("--decay", type=float, default=0.9) # decay for better收敛
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

    if not osp.isdir(f"{args.file_path}/demos/"):
        os.makedirs(f"{args.file_path}/demos/")

    # 单次选择运行（STOMP,STODI）的主函数
    demo(args=args)