import matplotlib


matplotlib.rcParams['font.family'] = 'Microsoft YaHei'  # 设置为微软雅黑字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体

import matplotlib.pyplot as plt
import numpy as np

x_range = [-5, 5]
y_range = [-5, 5]
def peaks(x_range,
          y_range,
          n=100,
          plot_method=111,
          xlabel='X',
          ylabel='Y',
          zlabel='Z',
          ifplot=1):
    # 数据准备
    x = np.linspace(x_range[0], x_range[1], n)  # x轴数据范围
    y = np.linspace(y_range[0], y_range[1], n)  # y轴数据范围
    x_mesh, y_mesh = np.meshgrid(x, y)  # 创建网格
    # z = np.sin(np.sqrt(x_mesh ** 2 + y_mesh ** 2))  # 曲面高度
    z = 3 * ((1 - x) ** 2) * np.exp(- x_mesh ** 2 - (y_mesh + 1) ** 2) - 10 * ((x_mesh / 5) - x_mesh ** 3 - y_mesh ** 5) \
        * np.exp(- x_mesh ** 2 - y_mesh ** 2) - (1/3) * np.exp(-(x_mesh + 1) ** 2 - y_mesh ** 2)
    # print(z)
    # 创建3D图形对象
    if ifplot:

        fig = plt.figure()
        ax = fig.add_subplot(plot_method, projection='3d')

        # 绘制3D曲面图
        ax.plot_surface(x_mesh, y_mesh, z, cmap='viridis')

        # 设置坐标轴标签
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        # 显示图形
        plt.show()
    return z