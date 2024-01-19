from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import os
import scipy
import scipy.io as scio

def compute_vorticity(u, v):
    # 数据处理过程中计算速度场的梯度
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    # 计算涡度
    vorticity = dv_dx - du_dy
    return vorticity


def read_data(file_path):
    df = pd.read_table(file_path, header=None, sep='         |,', engine='python', skipinitialspace=True)
    df.columns = df.iloc[0]
    # df.columns = re.split(r'\s+', str(df.iloc[0]))
    columns = str(df.columns).split(',')
    # columns
    df = df[1:]
    df = df.iloc[:, :9]
    return df


def collect_data(df):
    vort = df['   vorticity-mag'].astype(float)
    v_m = df['velocity-magnitude'].astype(float)
    v_a = df['  velocity-angle'].astype(float)
    x_grid = df['    x-coordinate'].astype(float)
    y_grid = df['    y-coordinate'].astype(float)
    # 计算速度的 x 和 y 分量
    vx = v_m * np.cos(v_a)
    vy = v_m * np.sin(v_a)
    return x_grid, y_grid, vx, vy, vort


def DataStorage(folder_path):
    # 遍历文件夹
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_list.append(file_path)
    print(len(file_list))
    # 遍历文件
    Vx = []
    Vy = []
    VortAll = []
    for i, file_name in enumerate(file_list):
        df = read_data(file_name)
        if i == 0:
            x_grid, y_grid, vx, vy, vort = collect_data(df)
            Vx = vx
            Vy = vy
            VortAll = vort
        else:
            _, _, vx, vy, vort = collect_data(df)
            Vx = np.column_stack((Vx, vx))
            Vy = np.column_stack((Vy, vy))
            VortAll = np.column_stack((VortAll, vort))
    return x_grid, y_grid, Vx, Vy, VortAll


def data2matrix(x_grid, y_grid, data):
    target_x, target_y = np.meshgrid(
        np.arange(0, 0.095, 0.001),
        np.arange(0, 0.042, 0.001),
    )
    # 生成坐标点对
    points = np.column_stack((x_grid, y_grid))
    for i in range(data.shape[1]):  # 按列循环
        if i == 0:
            matrixAll = griddata(points, data[:, i], (target_x, target_y), method='nearest')
        else:
            matrix = griddata(points, data[:, i], (target_x, target_y), method='nearest')
            matrixAll = np.dstack((matrixAll, matrix))

    return matrixAll


def computeMatrixSpeed2Vort(U, V):
    for i in range(U.shape[2]):
        if i == 0:
            computedVort = compute_vorticity(U[:, :, i], V[:, :, i])
        else:
            computedVort = np.dstack((computedVort, compute_vorticity(U[:, :, i], V[:, :, i])))
    return computedVort


def toReal(x):
    if np.iscomplexobj(x):
        return float(np.real(x))


def matrix2real(matrix):
    return np.vectorize(toReal)(matrix)


def Uxyt2Uxt(Uxyt):
    # 把3维矩阵xyt压缩为xt
    [Ny, Nx, Nt] = [np.size(Uxyt, 0), np.size(Uxyt, 1), np.size(Uxyt, 2)]
    Nxy = Ny * Nx
    Uxt = np.reshape(Uxyt, (Nxy, Nt))
    return Uxt


def UV2UxyVxy(UVx, Ny, Nx):
    print("UVx:", UVx)
    print("Ny*Nx:", Ny * Nx)
    print("UVx:", len(UVx))
    Ux = UVx[0:Ny * Nx]
    print("Ux:", len(Ux))
    Vx = UVx[Ny * Nx::]
    print("Vx:", len(Vx))
    print(Vx)
    Uxy = np.reshape(Ux, (Ny, Nx))
    Vxy = np.reshape(Vx, (Ny, Nx))
    return Uxy, Vxy


def pod_svd(Utx):
    # 基于SVD的POD （在N远大于m时可加快速度节省内存）
    # 输入Utx，其中时间离散长度N = np.size(Utx, 1)， 空间离散长度m = np.size(Utx, 2)
    # 输出U0x， 0阶模态， 可以看作定长平均值
    # 输出An， 时间变量，对应模态的幅值随时间的变化，可以用来做时间序列分析
    # 输出phiU， POD模态
    # 输出Ds， 特征值Ds反映了每一个模态对应的能量，可以用来排序
    N = np.size(Utx, 0)  # 时间尺度
    m = np.size(Utx, 1)  # 空间尺度

    # 0阶
    U0x = np.mean(Utx, 0) # 空间平均

    Utx = Utx - (U0x * np.ones(N).reshape(N, 1)) * np.ones((N, m))
    # SVD分解，使用econ加速
    U, S, PhiU = np.linalg.svd(Utx, full_matrices=False)   # phiU是基函数，也是POD模态
    # 利用特征值和特征向量的方式进行POD分解
    print("size U:", np.size(U, 0), np.size(U, 1))
    print("size S:", np.size(S))
    print("size PhiU:", np.size(PhiU, 0), np.size(PhiU, 1))

    Ds = np.diag(S) ** 2 / N
    S = np.diag(S)
    An = U @ S
    print(S)
    # print(Ds)
    print("size An:", np.size(An, 0), np.size(An, 1))
    PhiU = PhiU.T
    return [U0x, An, PhiU, Ds]


def DMD_class(X, Y):
    # DMD经典算法
    # 输入变量X, Y分别为时空矩阵Uxt的1~N-1列以及2~N列
    # 输出变量Dd为经过DMD分解后排序过的特征值
    # 输出变量b为模态对应的初始结果，与模态相乘后可得初始结果Ux1
    # 输出变量Time_DMD为分解后的时间序列（已排序）
    # 输出变量Phi为DMD分解后的模态结果（已排序）
    # 输出变量Energy为DMD分解后的模态能力值（已排序）
    N = X.shape[1]
    # Step1 对X进行svd分解
    U, S, VT = scipy.linalg.svd(X, full_matrices=False)
    Sd = np.diag(S)
    r = np.sum(S > 1e-5)
    U = U[:, 0:r]
    Smatrix = Sd[0:r, 0:r]
    V = VT.T[:, 0:r]

    # Step2 求解转换矩阵A
    A = U.T @ Y @ V @ scipy.linalg.inv(Smatrix)
    # Step3 求解矩阵A的特征向量及特征值
    D, Om = scipy.linalg.eig(A)
    Dd = np.diag(D)
    # Step4 求DMD模态
    Phi = Y @ V @ scipy.linalg.inv(Smatrix) @ Om
    # Step5 计算模态对应的初始值b
    b, residuals, rank, singular_values = np.linalg.lstsq(Phi, X[:, 0], rcond=None)

    # Step6 模态排序
    Q = np.vander(D, N, increasing=True)  # 建立范德蒙矩阵来储存特征值变换
    Time_DMD = b.reshape(len(b), 1) * Q  # 获取模态对应的时间系数

    Energy = np.zeros(np.size(Phi, 1))
    for i in range(np.size(Phi, 1)):
        Uxt_DMD_k = np.real(Phi[:, i].reshape((Phi.shape[0], 1)) * Time_DMD[i, :].reshape((1, Time_DMD.shape[1])))
        E_k = np.sum((Uxt_DMD_k ** 2))
        Energy[i] = E_k

    # 对于每个模态的能量进行排序
    Ie = np.argsort(-Energy)
    Energy = Energy[Ie]
    Dd = Dd[Ie]
    b = b[Ie]
    Phi = Phi[:, Ie]
    Time_DMD = Time_DMD[Ie, :]

    return Dd, b, Phi, Time_DMD, Energy


def curl(Uxy0, Vxy0):
    return (np.gradient(Uxy0, axis=0) / 80 + np.gradient(Vxy0, axis=1) / 50)


def displayPOD2D_Vector(UV0x, PhiUV, Ds, x, y):
    X, Y = np.meshgrid(x, y)
    Nx = len(x)
    Ny = len(y)
    fig = plt.figure()

    ax0 = fig.add_axes([0.08,0.79,0.45,0.16])
    Uxy0, Vxy0 = UV2UxyVxy(UV0x, Ny, Nx)
    ax0.pcolor(X, Y, curl(Uxy0, Vxy0))
    ax0.quiver(X[::5, ::5],
               Y[::5, ::5],
               Uxy0[::5, ::5],
               Vxy0[::5, ::5],
               color="k")
    ax0.set_title("average", fontsize=8)

    ax1 = fig.add_axes([0.08,0.54,0.45,0.16])
    Uxy1, Vxy1 = UV2UxyVxy(PhiUV[:, 0], Ny, Nx)
    ax1.pcolor(X, Y, curl(Uxy1, Vxy1))
    k = 5
    ax1.quiver(X[::k, ::k],
               Y[::k, ::k],
               Uxy1[::k, ::k],
               Vxy1[::k, ::k],
               color="k",
               scale=0.8)
    ax1.set_title("Mode1", fontsize=8)

    ax2 = fig.add_axes([0.08,0.29,0.45,0.16])
    Uxy2, Vxy2 = UV2UxyVxy(PhiUV[:, 1], Ny, Nx)
    ax2.pcolor(X, Y, curl(Uxy2, Vxy2))
    k = 5
    ax2.quiver(X[::k, ::k],
               Y[::k, ::k],
               Uxy2[::k, ::k],
               Vxy2[::k, ::k],
               color="k",
               scale=0.8)
    ax2.set_title("Mode2", fontsize=8)

    ax3 = fig.add_axes([0.08,0.04,0.45,0.16])
    Uxy3, Vxy3 = UV2UxyVxy(PhiUV[:, 2], Ny, Nx)
    ax3.pcolor(X, Y, curl(Uxy3, Vxy3))
    k = 5
    ax3.quiver(X[::k, ::k],
               Y[::k, ::k],
               Uxy2[::k, ::k],
               Vxy2[::k, ::k],
               color="k",
               scale=0.8)
    ax3.set_title("Mode3", fontsize=8)

    # 能量分布
    # 计算归一化频率
    Ds_N = Ds / np.sum(Ds)
    Ds_N = np.sum(Ds_N, axis=1)
    if np.iscomplex(Ds_N).any():
        Ds_N = matrix2real(Ds_N)
    # 计算累计频率
    cum_Ds_N = np.cumsum(Ds_N)

    N_Cum = 10
    bar_width = 1

    ax5 = fig.add_axes([0.6,0.79,0.3,0.17])
    # ax5.bar(np.arange(1, N_Cum+1), cum_Ds_N[:N_Cum], width=bar_width, edgecolor='black')
    ax5.bar(np.arange(1, N_Cum+1), Ds_N[:N_Cum], width=bar_width, edgecolor='black')

    ax6 = fig.add_axes([0.65,0.54,0.3,0.17])
    # print("Ds_N[1]:", Ds_N[1])
    ax6.pie([Ds_N[0], 1-Ds_N[0]], wedgeprops={'width': 0.5},
                   autopct='%.1f%%')
    ax6.set_title("Mode1 Energy Ratio", fontsize=8)

    ax7 = fig.add_axes([0.65,0.30,0.3,0.17])
    # print("Ds_N[1]:", Ds_N[1])
    ax7.pie([Ds_N[1], 1 - Ds_N[1]], wedgeprops={'width': 0.5},
            autopct='%.1f%%')
    ax7.set_title("Mode2 Energy Ratio", fontsize=8)

    ax8 = fig.add_axes([0.65,0.05,0.3,0.17])
    # print("Ds_N[1]:", Ds_N[1])
    ax8.pie([Ds_N[2], 1 - Ds_N[2]], wedgeprops={'width': 0.5},
            autopct='%.1f%%')
    ax8.set_title("Mode3 Energy Ratio", fontsize=8)

    plt.show()

    return 0

