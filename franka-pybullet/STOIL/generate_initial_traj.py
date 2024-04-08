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