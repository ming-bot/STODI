import sys
sys.path.append(r'E:/Proud/franka-pybullet/src')
import argparse
import numpy as np
from robot_model import Panda
from calculate_cost import Multi_Cost
from main import generate_multi_state
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def Draw_multi_contourCost(cost_list):
    x = np.linspace(0, 1, cost_list.shape[0])
    labels = ['kl', 'nmse', 'mse', 'dtw', 'js']
    y1 = cost_list[:, 0]
    y2 = cost_list[:, 1]
    y3 = cost_list[:, 2]
    y4 = cost_list[:, 3]
    y5 = cost_list[:, 4]
    plt.plot(x, 25000 * y1, label=labels[0], c='red', linewidth=1)
    plt.plot(x, 100000 * y2, label=labels[1], c='blue', linewidth=1)
    plt.plot(x, y3 / 5, label=labels[2], c='green', linewidth=1)
    plt.plot(x, 1.5 * y4, label=labels[3], c='black', linewidth=1)
    plt.plot(x, 160000 * y5, label=labels[4], c='orange', linewidth=1)

    plt.legend()
    plt.show()


if __name__ == '__main__':
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

    path = f'{args.file_path}/results/{args.expt_name}/joints_logs.txt'
    # joints_data = np.loadtxt(path, delimiter=' ') # m * (256 * 7)
    file = open(path, "r")
    joints_data = file.readlines()
    file.close()
    for i in range(len(joints_data)):
        joints_data[i] = joints_data[i].split()
    
    joints_data = np.array(joints_data).astype(float)
    # print(joints_data.shape)

    robot = Panda()
    cost_function = Multi_Cost(panda=robot, init_trajectory=joints_data[0, :].reshape((-1,7)), args=args)
    
    contour_cost = []
    
    for i in range(joints_data.shape[0]):
        joints = joints_data[i, :].reshape((-1, 7))
        # print(joints.shape)
        cost_function.Update_state(generate_multi_state(joints, args=args))
        contour_cost.append([cost_function.calculate_contour_cost('kl'),
                             cost_function.calculate_contour_cost('nmse'),
                             cost_function.calculate_contour_cost('mse'),
                             cost_function.calculate_contour_cost('dtw'),
                             cost_function.calculate_contour_cost('js')
        ])
    contour_cost = np.array(contour_cost)
    print(contour_cost.shape)

    Draw_multi_contourCost(contour_cost)

    # t_stat, p_value = stats.ttest_ind(25000 * contour_cost[:, 0], 100000 * contour_cost[:, 1])
    # print("kl,nmse:", t_stat, p_value)
    # data = {"kl": 25000 * contour_cost[:, 0],
    # "nmse": 100000 * contour_cost[:, 1]}
    # df = pd.DataFrame(data)
    # pearson_corr = df['kl'].corr(df['nmse'])
    # print(f"Pearson相关系数(kl,nmse): {pearson_corr}")

    # t_stat, p_value = stats.ttest_ind(25000 * contour_cost[:, 0], contour_cost[:, 2] / 4)
    # print("kl,mse:", t_stat, p_value)
    # data = {"kl": 25000 * contour_cost[:, 0],
    # "mse": contour_cost[:, 2] / 4}
    # df = pd.DataFrame(data)
    # pearson_corr = df['kl'].corr(df['mse'])
    # print(f"Pearson相关系数(kl,mse): {pearson_corr}")
    
    # t_stat, p_value = stats.ttest_ind(25000 * contour_cost[:, 0], contour_cost[:, 3])
    # print("kl,dtw:", t_stat, p_value)
    # data = {"kl": 25000 * contour_cost[:, 0],
    # "dtw": contour_cost[:, 3]}
    # df = pd.DataFrame(data)
    # pearson_corr = df['kl'].corr(df['dtw'])
    # print(f"Pearson相关系数(kl,dtw): {pearson_corr}")

    # t_stat, p_value = stats.ttest_ind(100000 * contour_cost[:, 1], contour_cost[:, 2] / 4)
    # print("nmse,mse:", t_stat, p_value)
    # data = {"nmse": 100000 * contour_cost[:, 1],
    # "mse": contour_cost[:, 2] / 4}
    # df = pd.DataFrame(data)
    # pearson_corr = df['nmse'].corr(df['mse'])
    # print(f"Pearson相关系数(nmse,mse): {pearson_corr}")

    # t_stat, p_value = stats.ttest_ind(100000 * contour_cost[:, 1], contour_cost[:, 3])
    # print("nmse,dtw:", t_stat, p_value)
    # data = {"nmse": 100000 * contour_cost[:, 1],
    # "dtw": contour_cost[:, 3]}
    # df = pd.DataFrame(data)
    # pearson_corr = df['nmse'].corr(df['dtw'])
    # print(f"Pearson相关系数(nmse,dtw): {pearson_corr}")

    # t_stat, p_value = stats.ttest_ind(contour_cost[:, 2] / 4, contour_cost[:, 3])
    # print("mse,dtw:", t_stat, p_value)
    # data = {"mse": contour_cost[:, 2] / 4,
    # "dtw": contour_cost[:, 3]}
    # df = pd.DataFrame(data)
    # pearson_corr = df['mse'].corr(df['dtw'])
    # print(f"Pearson相关系数(mse.dtw): {pearson_corr}")

    # t_stat, p_value = stats.ttest_ind(160000 * contour_cost[:, 4], contour_cost[:, 3])
    # print("js,dtw:", t_stat, p_value)
    # data = {"js": 160000 * contour_cost[:, 4],
    # "dtw": contour_cost[:, 3]}
    # df = pd.DataFrame(data)
    # pearson_corr = df['js'].corr(df['dtw'])
    # print(f"Pearson相关系数(js,dtw): {pearson_corr}")
