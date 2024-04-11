import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def Draw_trajectory(init_traj, trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [point[0] for point in trajectory]
    y = [point[1] for point in trajectory]
    z = [point[2] for point in trajectory]

    ax.scatter(x, y, z, c='red')

    x = [point[0] for point in init_traj]
    y = [point[1] for point in init_traj]
    z = [point[2] for point in init_traj]

    ax.scatter(x, y, z, c='blue')

    ax.set_xlabel('X')
    ax.set_xlabel('Y')
    ax.set_xlabel('Z')

    plt.show()