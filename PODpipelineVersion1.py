from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from libVersion1 import *
# 1805数据集的数据点生成方式存疑，其坐标位置并非设定的坐标点，怀疑是输出的区域内网格中心点坐标
# 本方法使用griddata将数据点向网格重聚类形成数据矩阵
x_grid, y_grid, Vx, Vy, VortAll = DataStorage('dataset/0115FFF26')
Vort = data2matrix(x_grid, y_grid, VortAll)
U = data2matrix(x_grid, y_grid, Vx)
V = data2matrix(x_grid, y_grid, Vy)
computedVort = computeMatrixSpeed2Vort(U, V)
# 计算UV向量
U_xt = Uxyt2Uxt(U)  # 把二维问题转化为一维问题
V_xt = Uxyt2Uxt(V)

# 合并UV向量
UV_xt = np.vstack((U_xt, V_xt))
U0x, An, PhiU, Ds = pod_svd(UV_xt.T)

x_coords = np.arange(0, 0.095, 0.001)
y_coords = np.arange(0, 0.042, 0.001)
displayPOD2D_Vector(U0x, PhiU, Ds, x_coords, y_coords)
