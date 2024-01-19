import numpy as np
import matplotlib.pyplot as plt
# %matplotlib qt5
from matplotlib.animation import FuncAnimation
from scipy import signal
from classicPOD import pod_svd, matrix2real
def Uxyt2Uxt(Uxyt):
    # 把3维矩阵xyt压缩为xt
    [Ny, Nx, Nt] = [np.size(Uxyt, 0), np.size(Uxyt, 1), np.size(Uxyt, 2)]
    Nxy = Ny * Nx
    Uxt = np.reshape(Uxyt, (Nxy, Nt))
    return Uxt


def UV2UxyVxy(UVx, Ny, Nx):
    print("UVx:", UVx)
    print("Ny*Nx:", Ny*Nx)
    print("UVx:", len(UVx))
    Ux = UVx[0:Ny*Nx]
    print("Ux:", len(Ux))
    Vx = UVx[Ny*Nx::]
    print("Vx:", len(Vx))
    print(Vx)
    Uxy = np.reshape(Ux, (Ny, Nx))
    Vxy = np.reshape(Vx, (Ny, Nx))
    return Uxy, Vxy


def curl(Uxy0, Vxy0, X, Y):

    return (np.gradient(Uxy0, axis=0) / 80 + np.gradient(Vxy0, axis=1) / 50)

def displayPOD2D(U0x, PhiU, Ds, x, y):
    X, Y = np.meshgrid(x, y)
    Nx = len(x)
    Ny = len(y)
    fig = plt.figure(figsize=(9, 7))

    # 4模态
    ax1 = fig.add_axes([0.08, 0.80, 0.45, 0.16])  # 平均0阶
    Uxy0 = np.reshape(U0x, (Ny, Nx))  # 还原0阶信号
    ax1.pcolor(X, Y, Uxy0)
    ax1.set_title("0阶模态/平均值", fontsize=8)

    ax2 = fig.add_axes([0.08,0.57,0.45,0.16])  # 1阶模态
    Uxy1 = np.reshape(PhiU[:, 0], (Ny, Nx))
    ax2.pcolor(X, Y, Uxy1)
    ax2.set_title("1阶模态", fontsize=8)

    ax3 = fig.add_axes([0.08, 0.34, 0.45, 0.16])  # 2阶模态
    Uxy2 = np.reshape(PhiU[:, 1], (Ny, Nx))
    ax3.pcolor(X, Y, Uxy2)
    ax3.set_title("2阶模态", fontsize=8)

    ax4 = fig.add_axes([0.08, 0.11, 0.45, 0.16])  # 3阶模态
    Uxy3 = np.reshape(PhiU[:, 2], (Ny, Nx))
    ax4.pcolor(X, Y, Uxy3)
    ax4.set_title("3阶模态", fontsize=8)


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
    ax5.bar(np.arange(1, N_Cum+1), cum_Ds_N[:N_Cum], width=bar_width, edgecolor='black')

    ax6 = fig.add_axes([0.65,0.54,0.3,0.17])
    # print("Ds_N[1]:", Ds_N[1])
    ax6.pie([Ds_N[0], 1-Ds_N[0]], wedgeprops={'width': 0.5},
                   autopct='%.1f%%')
    ax6.set_title("1阶模态能量占比", fontsize=8)

    ax7 = fig.add_axes([0.65,0.30,0.3,0.17])
    # print("Ds_N[1]:", Ds_N[1])
    ax7.pie([Ds_N[1], 1 - Ds_N[1]], wedgeprops={'width': 0.5},
            autopct='%.1f%%')
    ax7.set_title("2阶模态能量占比", fontsize=8)

    ax8 = fig.add_axes([0.65,0.05,0.3,0.17])
    # print("Ds_N[1]:", Ds_N[1])
    ax8.pie([Ds_N[2], 1 - Ds_N[2]], wedgeprops={'width': 0.5},
            autopct='%.1f%%')
    ax8.set_title("3阶模态能量占比", fontsize=8)

    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    plt.show()

def scalaPOD():
    # 初始化
    x = np.arange(-6, 6.2, 0.2)  # 61
    y = np.arange(-5, 5.2, 0.2)  # 51
    t = np.arange(0, 6.05, 0.05)  # 121
    [X, Y, T] = np.meshgrid(x, y, t)
    print(np.size(X, 0),
          np.size(X, 1),
          np.size(X, 2))
    [Ny, Nx, Nt] = [np.size(X, 0), np.size(X, 1), np.size(X, 2)]

    # 自定义流场，U是沿x方向的速度分量，v是y方向的速度分量
    print("Y:", Y)
    U0 = -1 * Y ** 2 + 25
    print("U0:", U0)
    V0 = 0 * Y
    # U0 V0不随时间变化，设为定常流场
    U1 = -5 * np.sin(Y) * np.cos(2 * np.pi / 20 * Y) * (0.1 + np.exp(T / 10))
    V1 = 5 * np.sin(X - T) * np.cos(2 * np.pi / 20 * Y) * (0.1 + np.exp(T / 10))
    U2 = -1 * np.cos( 2 * Y) * np.cos(2 * np.pi / 20 * Y) * (0.2 + np.exp(- T / 3))
    V2 = 1 * np.cos(2 * (X - 3 * T)) * np.cos(2 * np.pi / 20 * Y) * (0.2 + np.exp(- T / 3))
    print("U1:", U1)
    print("V1:", V1)
    print("U2:", U2)
    print("V2:", V2)

    U_Sum = U0 + U1 + U2
    V_Sum = V0 + V1 + V2

    P = 1 + np.sqrt(U0 ** 2 + V0 ** 2) + np.sqrt(U1 ** 2 + V1 ** 2) + np.sqrt(U2 ** 2 + V2 ** 2)
    # 示范标量，不代表任何物理意义
    print("P:", P)
    # 计算变量P

    P_xt = Uxyt2Uxt(P)
    print("P_xt:", P_xt)

    U0x, An, PhiU, Ds = pod_svd(P_xt.T)
    # POD处理tx格式，需要对xt数据转置
    # displayPod2(U0x, PhiU, Ds, x, y)
    displayPOD2D(U0x, PhiU, Ds, x, y)



def displayPOD2D_Vector(UV0x, PhiUV, Ds, x, y):
    X, Y = np.meshgrid(x, y)
    Nx = len(x)
    Ny = len(y)
    fig = plt.figure()

    ax0 = fig.add_axes([0.08,0.79,0.45,0.16])
    Uxy0, Vxy0 = UV2UxyVxy(UV0x, Ny, Nx)
    ax0.pcolor(X, Y, curl(Uxy0, Vxy0, X, Y))
    ax0.quiver(X[::5, ::5],
               Y[::5, ::5],
               Uxy0[::5, ::5],
               Vxy0[::5, ::5],
               color="k")
    ax0.set_title("0阶模态/平均值", fontsize=8)

    ax1 = fig.add_axes([0.08,0.54,0.45,0.16])
    Uxy1, Vxy1 = UV2UxyVxy(PhiUV[:, 0], Ny, Nx)
    ax1.pcolor(X, Y, curl(Uxy1, Vxy1, X, Y))
    k = 5
    ax1.quiver(X[::k, ::k],
               Y[::k, ::k],
               Uxy1[::k, ::k],
               Vxy1[::k, ::k],
               color="k",
               scale=0.8)
    ax1.set_title("1阶模态", fontsize=8)

    ax2 = fig.add_axes([0.08,0.29,0.45,0.16])
    Uxy2, Vxy2 = UV2UxyVxy(PhiUV[:, 1], Ny, Nx)
    ax2.pcolor(X, Y, curl(Uxy2, Vxy2, X, Y))
    k = 5
    ax2.quiver(X[::k, ::k],
               Y[::k, ::k],
               Uxy2[::k, ::k],
               Vxy2[::k, ::k],
               color="k",
               scale=0.8)
    ax2.set_title("2阶模态", fontsize=8)

    ax3 = fig.add_axes([0.08,0.04,0.45,0.16])
    Uxy3, Vxy3 = UV2UxyVxy(PhiUV[:, 2], Ny, Nx)
    ax3.pcolor(X, Y, curl(Uxy3, Vxy3, X, Y))
    k = 5
    ax3.quiver(X[::k, ::k],
               Y[::k, ::k],
               Uxy2[::k, ::k],
               Vxy2[::k, ::k],
               color="k",
               scale=0.8)
    ax3.set_title("3阶模态", fontsize=8)

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
    ax5.bar(np.arange(1, N_Cum+1), cum_Ds_N[:N_Cum], width=bar_width, edgecolor='black')

    ax6 = fig.add_axes([0.65,0.54,0.3,0.17])
    # print("Ds_N[1]:", Ds_N[1])
    ax6.pie([Ds_N[0], 1-Ds_N[0]], wedgeprops={'width': 0.5},
                   autopct='%.1f%%')
    ax6.set_title("1阶模态能量占比", fontsize=8)

    ax7 = fig.add_axes([0.65,0.30,0.3,0.17])
    # print("Ds_N[1]:", Ds_N[1])
    ax7.pie([Ds_N[1], 1 - Ds_N[1]], wedgeprops={'width': 0.5},
            autopct='%.1f%%')
    ax7.set_title("2阶模态能量占比", fontsize=8)

    ax8 = fig.add_axes([0.65,0.05,0.3,0.17])
    # print("Ds_N[1]:", Ds_N[1])
    ax8.pie([Ds_N[2], 1 - Ds_N[2]], wedgeprops={'width': 0.5},
            autopct='%.1f%%')
    ax8.set_title("3阶模态能量占比", fontsize=8)

    plt.show()

    return 0



def vectorPOD():
    # 初始化
    x = np.arange(-8, 8.2, 0.2)  # 81
    y = np.arange(-5, 5.2, 0.2)  # 51
    t = np.arange(0, 6.05, 0.05)  # 121
    [X, Y, T] = np.meshgrid(x, y, t)
    print(np.size(X, 0),
          np.size(X, 1),
          np.size(X, 2))
    [Ny, Nx, Nt] = [np.size(X, 0), np.size(X, 1), np.size(X, 2)]

    # 自定义流场，U是沿x方向的速度分量，v是y方向的速度分量
    print("Y:", Y)
    U0 = -1 * Y ** 2 + 25
    print("U0:", U0)
    V0 = 0 * Y
    # U0 V0不随时间变化，设为定常流场
    U1 = -5 * np.sin(Y) * np.cos(2 * np.pi / 20 * Y) * (0.1 + np.exp(T / 10))
    V1 = 5 * np.sin(X - T) * np.cos(2 * np.pi / 20 * Y) * (0.1 + np.exp(T / 10))
    U2 = -1 * np.cos(2 * Y) * np.cos(2 * np.pi / 20 * Y) * (0.2 + np.exp(- T / 3))
    V2 = 1 * np.cos(2 * (X - 3 * T)) * np.cos(2 * np.pi / 20 * Y) * (0.2 + np.exp(- T / 3))
    print("U1:", U1)
    print("V1:", V1)
    print("U2:", U2)
    print("V2:", V2)

    U_Sum = U0 + U1 + U2
    V_Sum = V0 + V1 + V2
    # 计算UV向量
    U_xt = Uxyt2Uxt(U_Sum)  # 把二维问题转化为一维问题
    V_xt = Uxyt2Uxt(V_Sum)


    # 合并UV向量
    UV_xt = np.vstack((U_xt, V_xt))
    U0x, An, PhiU, Ds = pod_svd(UV_xt.T)
    # POD处理tx格式，需要对xt数据转置
    # displayPod2(U0x, PhiU, Ds, x, y)
    displayPOD2D_Vector(U0x, PhiU, Ds, x, y)
