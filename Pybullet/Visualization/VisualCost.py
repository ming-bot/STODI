import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import copy

palette = pyplot.get_cmap('Set1')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 32,
}

def draw_line(iters, name_of_alg, color_index, datas):
    color=palette(color_index)
    avg=np.mean(datas,axis=0)
    std=np.std(datas,axis=0)
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))#上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))#下方差
    plt.plot(iters, avg, color=color,label=name_of_alg,linewidth=3.5)
    plt.fill_between(iters, r1, r2, color=color, alpha=0.2)

def Draw_Cost(STOMP_COST, STODI_COST, TIME):
    sample_num = 50
    fig = plt.figure(figsize=(20,10))
    # 生成横坐标时间列表
    iters = np.linspace(0, TIME, num=sample_num + 1)

    stomp_data = []
    for raw_cost in STOMP_COST:
        # print(len(raw_cost))
        selected_points = np.linspace(0, len(raw_cost) - 1, sample_num+1, dtype=int)
        stomp_data.append(np.array(raw_cost)[selected_points, 1])
    
    stodi_data = []
    for raw_cost in STODI_COST:
        # print(len(raw_cost))
        selected_points = np.linspace(0, len(raw_cost) - 1, sample_num+1, dtype=int)
        stodi_data.append(np.array(raw_cost)[selected_points, 1])
    
    draw_line(iters, "STOMP", 1, stomp_data)
    draw_line(iters, "STODI", 2, stodi_data)

    # 设置横坐标范围和标签
    plt.xticks(np.arange(0, TIME + 1, step=10), fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Time(s)', fontsize=32)
    plt.ylabel('Cost', fontsize=32)
    plt.legend(loc='upper left', prop=font1)
    plt.title("Stochastic Optimization Cost", fontsize=34)

    plt.show()