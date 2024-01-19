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

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig)
'''
fig = plt.figure(figsize=(12, 8))
ax = fig.gca(projection='3d')
'''

delta = 0.125
# 生成代表X轴数据的列表
x = np.arange(-3.0, 3.0, delta)
# 生成代表Y轴数据的列表
y = np.arange(-2.0, 2.0, delta)
# 对x、y数据执行网格化
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# 计算Z轴数据（高度数据）
Z = (Z1 - Z2) * 2
# 绘制3D图形
surf = ax.plot_surface(X, Y, Z,
                       rstride=1,  # rstride（row）指定行的跨度
                       cstride=1,  # cstride(column)指定列的跨度
                       cmap=plt.get_cmap('rainbow'))  # 设置颜色映射
# 设置Z轴范围
ax.set_zlim(-2, 2)
# 设置标题
plt.title("3D图")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
