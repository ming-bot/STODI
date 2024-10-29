import numpy as np

# 在关节空间直接连接起点和终点，函数输入：起点和终点的关节角度list
def Joint_linear_initial(begin, end):
    # begin point should be 1 * 6, the same with end point
    begin = np.array(begin)
    end = np.array(end)
    points_num = 128
    initial_trajectory = np.zeros(shape=(points_num, 6))
    for i in range(points_num):
        initial_trajectory[i, :] = begin + (end - begin) * float(i)  / (points_num - 1)
    return initial_trajectory

# 函数输入同上，设置一个确定的中间点姿态，形成demonstration
def Generate_demonstration(begin, end):
    external = np.array([
            1.0,
            1.62,
            -1.12,
            0.47,
            0.22,
            0.56
        ])
    begin = np.array(begin)
    end = np.array(end)
    points_num = 64
    initial_trajectory = np.zeros(shape=(2 * points_num, 6))
    for i in range(points_num):
        initial_trajectory[i, :] = begin + (external - begin) * float(i)  / (points_num - 1)
        initial_trajectory[i + points_num, :] = external + (end - external) * float(i)  / (points_num - 1)
    return initial_trajectory

# 从末端执行器欧式空间轨迹得到关节空间轨迹
def Generate_demonstration_from_effector(effector_trajectory, robot):
    effector_trajectory = np.array(effector_trajectory)
    N = effector_trajectory.shape[0]
    joints_list = []
    current_init_effector = robot.get_current_end_effector()
    for i in range(N):
        if effector_trajectory.shape[1] == 3:
            pos = effector_trajectory[i, :]
            joints_list.append(robot.solveInverseKinematics(pos, current_init_effector[3:]))
        elif effector_trajectory.shape[1] == 7:
            pos = effector_trajectory[i, :2]
            ori = effector_trajectory[i, 3:]
            joints_list.append(robot.solveInverseKinematics(pos, ori))
        else:
            raise Exception('wrong joint trajectory: number of joints is not 7')
    return np.array(joints_list)

# 获取一些末端执行器在欧氏空间的轨迹（配合上个函数使用
def Generate_effector_trajectory(N, ep, shape, begin, inter, robot):
    temp = np.stack([np.array(begin),np.array(inter)], axis=0)
    print(temp.shape)
    begin_eff = robot.solveListKinematics(temp)[0,:]
    inter_eff = robot.solveListKinematics(temp)[1,:]
    if shape == 'circle':
        points_list = generate_circle_points(begin_eff[:3], inter_eff[:3], N)
    elif shape == 'linear':
        x = np.linspace(begin_eff[0], inter_eff[0], N)
        y = np.linspace(begin_eff[1], inter_eff[1], N)
        z = np.linspace(begin_eff[2], inter_eff[2], N)
        points_list = np.stack([x,y,z], axis=1)
        print(points_list.shape)

    trajectory = np.array(points_list)
    noise = np.random.normal(0, ep, size=(N, 3))
    # noise = np.random.rayleigh(scale=ep,size=(N,3))
    trajectory[1:-1] += noise[1:-1]
    return trajectory

# 获取一些三维圆周的list
def generate_circle_points(point1, point2, num_points):
    center = (point1 + point2) / 2
    radius = np.linalg.norm(point1 - point2) / 2
    normal = (point1 - point2) / np.linalg.norm(point1 - point2)
    
    # Generate points on a circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_points = np.zeros((num_points, 3))
    
    for i in range(num_points):
        x = center[0] + radius * np.cos(theta[i])
        y = center[1] + radius * np.sin(theta[i])
        z = center[2]
        
        # Rotate points to align with the vector between point1 and point2
        v = np.cross(normal, [0, 0, 1])
        c = np.dot(normal, [0, 0, 1])
        skew_matrix = np.array([[0, -v[2], v[1]],
                                [v[2], 0, -v[0]],
                                [-v[1], v[0], 0]])
        R = np.eye(3) + skew_matrix + np.dot(skew_matrix, skew_matrix) * (1 - c) / (np.linalg.norm(v)**2)
        point = np.dot(R, np.array([x - center[0], y - center[1], 0])) + center
        circle_points[i] = point
    
    return circle_points