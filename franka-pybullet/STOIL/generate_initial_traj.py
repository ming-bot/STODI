import numpy as np

def Joint_linear_initial(begin, end):
    # begin point should be 1 * 7, the same with end point
    begin = np.array(begin)
    end = np.array(end)
    points_num = 256
    initial_trajectory = np.zeros(shape=(points_num, 7))
    for i in range(points_num):
        initial_trajectory[i, :] = begin + (end - begin) * float(i)  / (points_num - 1)
    return initial_trajectory

def Generate_demonstration(begin, end):
    external = np.array([-0.3948859155257995, -0.7270775750246515, 0.9705156650028227, -1.7756625474152135, -0.2109975244307001, 2.7461024636268965, 1.071975144388428])
    begin = np.array(begin)
    end = np.array(end)
    points_num = 128
    initial_trajectory = np.zeros(shape=(2 * points_num, 7))
    for i in range(points_num):
        initial_trajectory[i, :] = begin + (external - begin) * float(i)  / (points_num - 1)
        initial_trajectory[i + points_num, :] = external + (end - external) * float(i)  / (points_num - 1)
    return initial_trajectory
