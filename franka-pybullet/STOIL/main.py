import sys
sys.path.append(r'E:/Proud/franka-pybullet/src')

import argparse
import os
import os.path as osp
import numpy as np
import copy
from MultiDimension import Multi_dimensions_stomp
from robot_model import Panda
from calculate_cost import Multi_Cost
import copy
import matplotlib.pyplot as plt
from generate_initial_traj import *

def Draw_cost(cost_list):
    x = np.linspace(0, 1, len(cost_list))
    y = cost_list
    plt.plot(x, y, linewidth=1)
    plt.show()

def generate_multi_state(n7_trajectory, args):
    velocity = np.zeros(shape=n7_trajectory.shape)
    acceleration = np.zeros(shape=n7_trajectory.shape)
    dt = 1.0 / args.sample_frequency
    
    for i in range(n7_trajectory.shape[0] - 1):
        acceleration[i, :] = (2.0 / dt ** 2) * (n7_trajectory[i + 1, :] - n7_trajectory[i, :] - velocity[i, :] * dt)
        velocity[i + 1, :] = velocity[i, :] + acceleration[i, :] * dt

    return {"position": copy.copy(n7_trajectory), "velocity": copy.copy(velocity), "acceleration": copy.copy(acceleration)}

def main(args):
    # 0. input initial trajectory; trajectory should be np.array((N * 7))
    initial_trajectory = Joint_linear_initial(begin=[0, 0, 0, -1.6, 0, 1.87, 0], end=[0, -0.7, 0, -1.6, 0, 3.5, 0.7])
    
    # 1. initial stomp
    robot = Panda()
    robot.setControlMode("position")
    cost_function = Multi_Cost(panda=robot, init_trajectory=initial_trajectory, args=args)
    stomp_panda = Multi_dimensions_stomp(num_points=initial_trajectory.shape[0], args=args, cost_func=cost_function, 
    rangelimit_low_list=np.array([-6.28,-6.28,-6.28,-6.28,-6.28,-6.28,-6.28]), rangelimit_high_list=np.array([6.28,6.28,6.28,6.28,6.28,6.28,6.28]), 
    vellimit_low_list=np.array([-50,-50,-50,-50,-50,-50,-50]), vellimit_high_list=np.array([50,50,50,50,50,50,50]),
    acclimit_low_list=np.array([-1000,-1000,-1000,-1000,-1000,-1000,-1000]), acclimit_high_list=np.array([1000,1000,1000,1000,1000,1000,1000]))

    stomp_panda.input_demonstration(initial_trajectory)

    Qcost_list = []
    Qcost_total_list = []
    iter_num = 0

    cost_function.Update_state(generate_multi_state(initial_trajectory, args=args))
    Qcost_list.append(cost_function.calculate_total_cost())

    iter_traj = copy.copy(initial_trajectory)
    temp_iter_traj = copy.copy(initial_trajectory)

    # begin optimal loops
    while(iter_num < 2000):
        iter_num += 1
        # generate noisy trajectory
        stomp_panda.multiDimension_diffusion()
        # calculate weights
        kn_weights = stomp_panda.calculate_weights()
        n7_proved_noise = stomp_panda.calculate_delta_noise(weights_p=kn_weights)
        # print(n7_proved_noise)
        # get new trajectory
        # print(iter_traj)
        # print(n7_proved_noise[1:-1, :] * (1.0 / args.sample_frequency)**2)
        if iter_num <= 500:
            temp_iter_traj[1:-1, :] = iter_traj[1:-1, :] + (args.decay**iter_num) * n7_proved_noise[1:-1, :] * (1.0 / args.sample_frequency)**4 # N * 7
        else:
            temp_iter_traj[1:-1, :] = iter_traj[1:-1, :] + (1.0 / iter_num) * n7_proved_noise[1:-1, :] * (1.0 / args.sample_frequency)**4 # N * 7
        # print(temp_iter_traj[1, :])
        # limit check
        if stomp_panda.multi_limit_check(temp_iter_traj):
            # update dimensions' trajectory
            stomp_panda.multi_update(joints_traj=temp_iter_traj)
            if args.reuse_state:
                stomp_panda.update_reuse_traj(generate_multi_state(temp_iter_traj, args=args))
            cost_function.Update_state(generate_multi_state(temp_iter_traj, args=args))
            Qcost_list.append(cost_function.calculate_total_cost())
            Qcost_total_list.append(np.sum(Qcost_list[-1]))
            iter_traj = temp_iter_traj
        # else:
        #     args.decay = args.decay * 0.5
        #     print(args.decay)
    
    # robot.traj_control(iter_traj)
    # print(Qcost_list)
    Draw_cost(Qcost_total_list)

    logfile = open(f"{args.file_path}/results/{args.expt_name}/result_logs.txt", 'w')
    logfile.write(str(iter_traj) + "\n")
    logfile.write("\n".join(str(item) for item in Qcost_list))
    logfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default=r'E:/Proud/franka-pybullet/src')
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