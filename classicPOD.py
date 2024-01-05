
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib qt5
from matplotlib.animation import FuncAnimation
from scipy import signal
def toReal(x):
    if np.iscomplexobj(x):
        return float(np.real(x))


def matrix2real(matrix):
    return np.vectorize(toReal)(matrix)


def PODclass():
    # 初始化
    x = np.arange(0, 10.02, 0.02)
    t = np.arange(0, 5.02, 0.02)
    Fs = 1 / (t[2] - t[1])
    # 用于计算采样频率（Sampling Frequency）的代码。
    # 其中 t(2)-t(1) 表示时间向量中相邻两个时间点的差值，即采样时间间隔。
    [X, T] = np.meshgrid(x, t)
    N = np.size(T, 1)  # 时间长度 251
    m = np.size(X, 0)  # 空间长度 501


    # print("X:", X)
    # print("T:", T)
    # 构建信号
    q = 1.6 * np.sin( 2 * np.pi * (X + 2 * T)) + 0.5 * np.sin( 2 * np.pi * (3 * X + 4 * T)) + 0.1

    # plt.plot(q)
    # plt.show()

    # POD分解
    [U0x, An, PhiU, Ds] = pod_svd(q)
    displayPod1(U0x, An, PhiU, Ds, x, t, q)
    # displayPod2(An, PhiU, Ds, x, t, 2)


def pod_origin(Utx):
    # 原始POD （在N远大于m时可加快速度节省内存）
    # 输入Utx，其中时间离散长度N = np.size(Utx, 1)， 空间离散长度m = np.size(Utx, 2)
    # 输出U0x， 0阶模态， 可以看作定长平均值
    # 输出An， 时间变量，对应模态的幅值随时间的变化，可以用来做时间序列分析
    # 输出phiU， POD模态
    # 输出Ds， 特征值Ds反映了每一个模态对应的能量，可以用来排序
    N = np.size(Utx, 0) # 时间尺度
    m = np.size(Utx, 1) # 空间尺度

    # 0阶
    U0x = np.mean(Utx, 0) # 空间平均
    # print('U0x:', U0x)
    # print(np.size(U0x))

    A = np.ones(N)
    A = A.reshape(A.shape[0], 1)
    # print('A:', A)
    A = U0x * A
    # print('A:', A)
    # print(np.size(A, 0), ';', np.size(A, 1))
    Utx = Utx - (A) * np.ones((N, m))
    # print('Utx:', Utx)
    # print(np.size(Utx, 0), ';', np.size(Utx, 1))
    # 利用特征值和特征向量的方式进行POD分解
    R = (1 / N) * Utx.T @ Utx
    D, PhiU = np.linalg.eig(R) # phiU是基函数，也是POD模态
    # print("D", D)
    # print("PhiU", PhiU)
    An = Utx @ PhiU
    # print("An", An)

    # 排序
    Ds = np.diag(D)
    # print(Ds)

    ind = np.argsort(-D) # 对D数组进行降序排列
    # print("ind:", ind)
    Ds = Ds[ind]
    print(matrix2real(Ds))
    # fig, axs = plt.subplots(1,1)
    # ax = axs[0, 1]
    # axs.pcolor(matrix2real(Ds))

    An = An[:, ind]
    # print("An:", An)
    PhiU = PhiU[:, ind]
    # print("PhiU", PhiU)
    # plt.show()
    return [U0x, An, PhiU, Ds]


def pod_snapshot(Utx):
    # 快照POD （在N远大于m时可加快速度节省内存）
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
    Uxt = Utx.T
    C = Uxt.T.dot(Uxt)
    print("size C:", np.size(C, 0), np.size(C, 1))
    D, AU = np.linalg.eig(C)   # phiU是基函数，也是POD模态
    # 利用特征值和特征向量的方式进行POD分解
    print("size AU:", np.size(AU, 0), np.size(AU, 1))
    # 排序
    Ds = np.diag(D)
    # print(Ds)
    print("size Ds:", np.size(Ds, 0), np.size(Ds, 1))
    ind = np.argsort(-D)  # 对D数组进行降序排列
    # print("ind:", ind)
    Ds = Ds[ind]
    # print(matrix2real(Ds))
    # fig, axs = plt.subplots(1,1)
    # ax = axs[0, 1]
    # axs.pcolor(matrix2real(Ds))

    AU = AU[:, ind]
    # print("An:", An)
    # 特征函数
    part1 = Uxt @ AU
    PhiU = np.matmul(part1, np.linalg.inv(np.sqrt(Ds)))
    # print("PhiU", PhiU)
    # plt.show()
    An = Uxt.T.dot(PhiU)
    return [U0x, An, PhiU, Ds]


def pod_svd(Utx):
    # 快照POD （在N远大于m时可加快速度节省内存）
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


def displayPod1(U0x, An, PhiU, Ds, x, t, Utx):
    # 动态演示模态叠加效果
    N = len(t)
    Limit_Y = [-2, 2]
    fig, axes = plt.subplots(3, 2)
    axesDict = {0: axes[0, 0], 1: axes[0, 1], 2: axes[1, 0], 3: axes[1, 1], 4: axes[2, 0], 5: axes[2, 1]}
    if np.iscomplex(An).any():
        An = matrix2real(An)
    if np.iscomplex(PhiU).any():
        PhiU = matrix2real(PhiU)

    def update(k):
        for i in range(6):

            axesDict[0].cla()
            axesDict[0].set_ylim(Limit_Y[0], Limit_Y[1])
            axesDict[0].plot(x, Utx[k, :])
            axesDict[0].set_title("原始数据")

            axesDict[1].cla()
            axesDict[1].set_ylim(Limit_Y[0], Limit_Y[1])
            axesDict[1].plot(x, U0x)
            axesDict[1].set_title("平均数据")

            axesDict[2].cla()
            axesDict[2].set_ylim(Limit_Y[0], Limit_Y[1])
            print('An:', An[k, 1])
            print('PhiU:', PhiU[:, 1].T)

            P1 = An[k, 1]*(PhiU[:, 1].T)
            P2= An[k, 2]*(PhiU[:, 2].T)
            axesDict[2].plot(x, U0x + P1 + P2)
            axesDict[2].set_title("0阶至2阶模态总和")

            axesDict[3].cla()
            axesDict[3].set_ylim(Limit_Y[0], Limit_Y[1])
            P3 = An[k, 3] * (PhiU[:, 3].T)
            P4 = An[k, 4] * (PhiU[:, 4].T)
            axesDict[3].plot(x, U0x + P1 + P2 + P3 + P4)
            axesDict[3].set_title("0阶至4阶模态总和")

            axesDict[4].cla()
            axesDict[4].set_ylim(Limit_Y[0], Limit_Y[1])
            P5 = An[k, 5] * (PhiU[:, 5].T)
            P6 = An[k, 6] * (PhiU[:, 6].T)
            axesDict[4].plot(x, U0x + P1 + P2 + P3 + P4 + P5 + P6)
            axesDict[4].set_title("0阶至6阶模态总和")

            axesDict[5].cla()
            axesDict[5].axis('off')
            N_En = 2
            if np.iscomplex(Ds).any():
                top2 = np.sum(matrix2real(Ds[0:N_En]), axis=1).tolist()
                rest = np.sum(matrix2real(sum(Ds) - sum(Ds[0:N_En]))).tolist()
            else:
                top2 = np.sum(Ds[0:N_En], axis=1).tolist()
                rest = np.sum(sum(Ds) - sum(Ds[0:N_En])).tolist()

            ax = fig.add_axes([0.65, 0.1, 0.2, 0.2])
            ax.pie([top2[0], top2[1], rest], colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)], wedgeprops={'width': 0.5},
                   autopct='%.1f%%')
            ax.set_title("1阶与2阶模态能量占比")
            pass


    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=500,
        init_func=None
    )



    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    plt.show()


def displayPod2(An, PhiU, Ds, x, t, Mode):
    Limit_Y = [-2, 2]
    fig, axes = plt.subplots(2, 2)
    plt.axis('off')
    plt.clf()
    ax = fig.add_axes([0.08, 0.58, 0.5, 0.35])  # 模态显示
    ax.plot(x, PhiU[:, Mode])
    title = str(Mode) + "阶模态"
    ax.set_title(title)

    ax = fig.add_axes([0.08, 0.10, 0.5, 0.35])  # 功率谱分析
    Fs = 1 / (t[2] - t[1])
    N = len(t)
    Limit_X = [-5, 5]
    ax.set_xlim(Limit_X[0], Limit_X[1])
    f, pxx = signal.welch(An[:, Mode], nperseg=N-1, noverlap=round(N/2), fs=Fs)
    ax.plot(f, pxx)
    title = str(Mode) + "阶模态时间功率谱"
    ax.set_title(title)

    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    plt.show()

