import sys
sys.path.append(r'G:/Proud/franka-pybullet/src')

import argparse
import os
import os.path as osp
import numpy as np
import copy
from MultiDimension import Multi_dimensions_stomp
from sentry import Sentry, generate_multi_state
from robot_model import Panda
from calculate_cost import Multi_Cost
import copy
import matplotlib.pyplot as plt
from generate_initial_traj import *
from visualization import Draw_trajectory, Draw_3trajectory

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

def main(args):
    # 0. input initial trajectory; trajectory should be np.array((N * 7))
    initial_trajectory = Joint_linear_initial(begin=[0, 0, 0, -1.6, 0, 1.87, 0], end=[0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    # 想要模仿的Demonstration
    demostrantion = Generate_demonstration(begin=[0, 0, 0, -1.6, 0, 1.87, 0], end=[0, -0.7, 0, -1.6, 0, 3.5, 0.7])

    # 1. initial stomp
    robot = Panda()
    # eff = Generate_effector_trajectory(128, 0, 'linear', [0, 0, 0, -1.6, 0, 1.87, 0], [0, -0.7, 0, -1.6, 0, 3.5, 0.7], robot)
    # initial_trajectory = Joint_linear_initial(begin=[0, 0, 0, -1.6, 0, 1.87, 0], end=[0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    # demostrantion = Generate_demonstration_from_effector(eff, robot)
    # print(initial_trajectory.shape, demostrantion.shape)

    init_end_effector = robot.solveListKinematics(initial_trajectory)
    # Cost Function
    cost_function = Multi_Cost(panda=robot, init_trajectory=demostrantion, args=args)
    # Sentry
    sentry = Sentry(initial_trajectory, cost_function, args)
    # 多维Stomp框架
    stomp_panda = Multi_dimensions_stomp(num_points=initial_trajectory.shape[0], args=args, cost_func=cost_function, 
    rangelimit_low_list=np.array([-6.28,-6.28,-6.28,-6.28,-6.28,-6.28,-6.28]), rangelimit_high_list=np.array([6.28,6.28,6.28,6.28,6.28,6.28,6.28]), 
    vellimit_low_list=np.array([-50,-50,-50,-50,-50,-50,-50]), vellimit_high_list=np.array([50,50,50,50,50,50,50]),
    acclimit_low_list=np.array([-1000,-1000,-1000,-1000,-1000,-1000,-1000]), acclimit_high_list=np.array([1000,1000,1000,1000,1000,1000,1000]))

    voyager_Qcost_list = []
    voyager_Qcost_total_list = []
    voyager_iter_joints = []

    neighbour_Qcost_list = []
    neighbour_Qcost_total_list = []
    neighbour_iter_joints = []

    iter_num = 0
    total_iter_num = 300

    # 计算Cost标准步骤(for example)
    cost_function.Update_state(generate_multi_state(initial_trajectory, args=args))
    voyager_Qcost_list.append(cost_function.calculate_total_cost('omse'))
    voyager_Qcost_total_list.append(np.sum(voyager_Qcost_list[-1]))
    neighbour_Qcost_list.append(cost_function.calculate_total_cost('omse'))
    neighbour_Qcost_total_list.append(np.sum(neighbour_Qcost_list[-1]))

    voyager_iter_joints.append(initial_trajectory)
    neighbour_iter_joints.append(initial_trajectory)

    # begin optimal loops
    while(iter_num < total_iter_num):
        iter_num += 1
        # 0. generate noisy trajectory
        stomp_panda.multiDimension_diffusion()
        # 1.1 calculate voyager weights
        temp_iter_traj = copy.copy(sentry.pioneer['voyager'].traj)
        stomp_panda.multi_update(joints_traj=sentry.pioneer['voyager'].traj)
        kn_weights = stomp_panda.calculate_weights()
        n7_proved_noise = stomp_panda.calculate_delta_noise(weights_p=kn_weights)

        # 1.2 get new trajectory
        temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + (args.decay**iter_num) * n7_proved_noise[1:-1, :] * (1.0 / args.sample_frequency)**4 # N * 7
        # 1.3 limit check
        if stomp_panda.multi_limit_check(temp_iter_traj):
            # 1.4 Update Voyager
            sentry.Update('voyager', temp_iter_traj)
            # 1.5 Using for reuse
            if args.reuse_state:
                stomp_panda.update_reuse_traj(generate_multi_state(temp_iter_traj, args=args))
            # Cost voyager cost( Just for visualize )
            cost_function.Update_state(generate_multi_state(temp_iter_traj, args=args))
            voyager_Qcost_list.append(cost_function.calculate_total_cost('omse'))
            voyager_Qcost_total_list.append(np.sum(voyager_Qcost_list[-1]))
            # Record voyager
            voyager_iter_joints.append(copy.copy(temp_iter_traj))
        
        # 2.1 calculate neighbour weights
        temp_iter_traj = copy.copy(sentry.pioneer['neighbour'].traj)
        stomp_panda.multi_update(joints_traj=sentry.pioneer['neighbour'].traj)
        kn_weights = stomp_panda.calculate_weights()
        n7_proved_noise = stomp_panda.calculate_delta_noise(weights_p=kn_weights)

        # 2.2 get new trajectory
        # temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + (args.decay**iter_num) * n7_proved_noise[1:-1, :] * (1.0 / args.sample_frequency)**4 # N * 7
        temp_iter_traj[1:-1, :] = temp_iter_traj[1:-1, :] + n7_proved_noise[1:-1, :] * (1.0 / args.sample_frequency)**4 # N * 7
        # 2.3 limit check
        if stomp_panda.multi_limit_check(temp_iter_traj):
            # 2.4 Update Neighbour
            sentry.Update('neighbour', temp_iter_traj)
            # 2.5 Using for reuse
            if args.reuse_state:
                stomp_panda.update_reuse_traj(generate_multi_state(temp_iter_traj, args=args))
            # Cost voyager cost( Just for visualize )
            cost_function.Update_state(generate_multi_state(temp_iter_traj, args=args))
            neighbour_Qcost_list.append(cost_function.calculate_total_cost('omse'))
            neighbour_Qcost_total_list.append(np.sum(neighbour_Qcost_list[-1]))
            # Record voyager
            neighbour_iter_joints.append(copy.copy(temp_iter_traj))
        
        # 3. Update the best
        sentry.Update('best')
        # 4. clock align the best and the neighbour
        if iter_num % (total_iter_num / 10):
            sentry.Update('clock')



    print("loop is over!")
    input()
    final_traj_state = generate_multi_state(sentry.pioneer['best'].traj, args)
    robot.traj_torque_control(final_traj_state["position"], final_traj_state["velocity"], final_traj_state["acceleration"])
    # robot.traj_control(iter_traj)
    # print(Qcost_list)
    # Draw_cost(voyager_Qcost_total_list)
    # Draw_cost(neighbour_Qcost_total_list)
    Draw_cost(list(np.array(neighbour_Qcost_list)[:,1]))


    end_effector = robot.solveListKinematics(sentry.pioneer['best'].traj)
    eff = np.array(robot.solveListKinematics(demostrantion))
    Draw_3trajectory(init_end_effector, end_effector, eff)
    # print(end_effector.shape)
    write_trajectory(end_effector[:, :3], f"{args.file_path}/results/{args.expt_name}/trajectory_logs.txt")
    write_joints(np.array(neighbour_iter_joints), f"{args.file_path}/results/{args.expt_name}/joints_logs.txt")

    logfile = open(f"{args.file_path}/results/{args.expt_name}/result_logs.txt", 'w')
    logfile.write("\n".join(str(item) for item in neighbour_Qcost_list))
    logfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default=r'G:/Proud/franka-pybullet/src')
    parser.add_argument("--expt-name", type=str, required=True)

    parser.add_argument("--dimension", type=int, default=7)
    parser.add_argument("--add-contourCost", type=bool, default=True)
    parser.add_argument("--reuse-state", type=bool, default=True)
    parser.add_argument("--reuse-num", type=int, default=10)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--decay", type=float, default=0.99)

    parser.add_argument("--control-frequency", type=float, default=10)
    parser.add_argument("--sample-frequency", type=float, default=20)
    
    args = parser.parse_args()

    if not osp.isdir(f"{args.file_path}/results/"):
        os.makedirs(f"{args.file_path}/results/")

    if not osp.isdir(f"{args.file_path}/results/{args.expt_name}/"):
        os.makedirs(f"{args.file_path}/results/{args.expt_name}/")
    
    main(args=args)