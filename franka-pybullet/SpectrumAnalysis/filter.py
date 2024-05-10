'''
This code is mainly for calculating the similarity between two trajectories, espacially in contour or in shape.
Input: two trajectories; list1, list2, [coor1, coor2, coor3, ..., coorN]
Fixed time step: dt
'''
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
from scipy.special import rel_entr

def Generate_trajectory(N, ep, shape):
    if shape == 'circle':
        x = np.cos(np.linspace(-np.pi/2, np.pi/2, N))
        y = np.sin(np.linspace(-np.pi/2, np.pi/2, N))
        z = np.zeros(N)
    elif shape == 'linear':
        x = np.linspace(-1, 1, N)
        y = np.zeros(N)
        z = np.zeros(N)
    elif shape == 's-shape':
        x = np.cos(np.linspace(-2*np.pi, 2*np.pi, N))
        y = np.linspace(-2, 2, N)
        z = np.zeros(N)
    trajectory = []
    for i in range(N):
        trajectory.append([x[i], y[i], z[i]])
    trajectory = np.array(trajectory)
    noise = np.random.normal(0, ep, size=(N, 3))
    trajectory[1:-1] += noise[1:-1]
    return trajectory

def FFT(trajectory, shift: bool):
    # 进行多元FFT变换
    fft_result = np.fft.fftn(trajectory)
    print(fft_result.shape)
    # 将零频率移动到中心位置
    if shift:
        fft_result = np.fft.fftshift(fft_result)
    return fft_result

def IFFT(freq, shift):
    if shift:
        freq = np.fft.ifftshift(freq)
    traj = np.fft.ifftn(freq)
    print(traj.shape)
    return traj

def plt_freq(freq):
    x = np.arange(np.shape(freq)[0])
    # 创建图表和子图
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    # 绘制第一个子图
    axs[0].plot(x, freq[:, 0], color='blue')
    axs[0].set_title('Sine Function')
    # 绘制第二个子图
    axs[1].plot(x, freq[:, 1], color='red')
    axs[1].set_title('Cosine Function')
    # 绘制第三个子图
    axs[2].plot(x, freq[:, 2], color='green')
    axs[2].set_title('Tangent Function')
    # 调整子图之间的间距
    plt.tight_layout()
    # 显示图形
    plt.show()

def plt_trajectory(traj):
    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制三维曲面图
    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2])
    # 设置坐标轴和标题
    ax.set_xlim((-2,2))
    ax.set_ylim((-2,2))
    ax.set_zlim((-2,2))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Surface Plot')
    plt.show()

# def Scaled_length(reference_distri, compare_distri):
#     # 原始数组的行数和列数
#     original_rows, original_cols = compare_distri.shape
#     # 插值为3x10数组
#     desired_rows, desired_cols = reference_distri.shape
#     # 定义插值函数
#     interp_func = RegularGridInterpolator((np.arange(original_rows), np.arange(original_cols)), compare_distri)
#     # 生成插值节点
#     new_rows_indices = np.linspace(0, original_rows - 1, desired_rows)
#     new_cols_indices = np.linspace(0, original_cols - 1, desired_cols)
#     grid_x, grid_y = np.meshgrid(new_rows_indices, new_cols_indices, indexing='ij')
#     new_indices = np.stack((grid_x, grid_y), axis=-1)
#     # 在新数组大小上进行插值
#     new_array = interp_func(new_indices)
#     return new_array

def supperess_noise(freq):
    # 抑制低幅值噪声
    for i in range(freq.shape[0]):
        for j in range(freq.shape[1]):
            # if i > 20 and i < freq.shape[0] - 20 and np.abs(freq[i, j]) < 20:
            if np.abs(freq[i, j]) < 20:
                freq[i, j] = freq[i, j] / 20
    return freq

def normization_noise(freq):
    power = np.abs(freq)
    for j in range(power.shape[1]):
        sum = np.max(power[:, j])
        factor = power[:, j] / sum
        freq[:, j] = factor * freq[:, j]
    return freq

def lower_filter(freq):
    for i in range(freq.shape[0]):
        for j in range(freq.shape[1]):
            if i > 10 and i < freq.shape[0] - 10 and np.abs(freq[i, j]) < 10:
                freq[i, j] = freq[i, j] / 20
    return freq


if __name__ == "__main__":
    isshift = False
    # trajectory1 = Generate_trajectory(128, 0.1, 'circle')
    # trajectory1 = Generate_trajectory(128, 0.1, 'linear')
    trajectory1 = Generate_trajectory(128, 0.1, 's-shape')
    plt_trajectory(traj=trajectory1)

    # calculate FFT
    trajFFT1 = FFT(trajectory1, isshift)
    plt_freq(np.abs(trajFFT1))
    # trajFFT1 = supperess_noise(trajFFT1)
    # trajFFT1 = normization_noise(trajFFT1)
    trajFFT1 = lower_filter(trajFFT1)
    plt_freq(np.abs(trajFFT1))
    trajiFFT1 = IFFT(trajFFT1, isshift)
    print(trajiFFT1.real)
    plt_trajectory(trajiFFT1.real)
