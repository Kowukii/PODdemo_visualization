import numpy as np
from fractions import Fraction
from drawing import peaks
from classicPOD import PODclass
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from multiDimension import scalaPOD, vectorPOD
# np.set_printoptions(formatter={'all': lambda x: str(Fraction(x).limit_denominator())})
# 结果分数显示设置
def firstMatrix():
    x11 = np.array([[-3 / 4, -1 / 4, -1 / 8]]).T
    x12 = np.array([[5 / 4, -1 / 4, -1 / 8]]).T
    x13 = np.array([[5 / 4, -1 / 4, 7 / 8]]).T
    x14 = np.array([[1 / 4, 7 / 4, -1 / 8]]).T
    x21 = np.array([[-3 / 4, -1 / 4, 7 / 8]]).T
    x22 = np.array([[-3 / 4, 3 / 4, -1 / 8]]).T
    x23 = np.array([[-3 / 4, -9 / 4, 7 / 8]]).T
    x24 = np.array([[1 / 4, 3 / 4, -17 / 8]]).T

    x = [x11, x12, x13, x14, x21, x22, x23, x24]

    # 列向量乘行向量

    R1 = 0
    for i in x:
        R1 += i * i.T;
    R = 1 / 8 * R1
    print("R:\n", R)
    # 计算矩阵R的行列式（只要行列式不等于0，就可以求特征值和特征向量）
    b = np.linalg.det(R)
    print("b:", b)
    # 特征值和特征向量
    c = np.linalg.eig(R)
    print("c:\n", c)
    # 特征值
    print("c[0]:\n", c[0])
    # 特征向量
    print("c[1]:\n", c[1])


def firstSVDcal():
    z = peaks([-5, 5], [-5, 5], ifplot=0)

    # nrows, ncols = 2, 2
    fig, axs = plt.subplots(2, 2)

    [U, S, V] = np.linalg.svd(z) # svd分解
    S = np.diag(S)

    ax = axs[0, 0]
    ax.pcolor(z)
    ax.set_title('image (z)')

    ax = axs[0, 1]
    K1 = 1
    U1 = U[:, 0:K1]
    S1 = S[0:K1, 0:K1]
    V1 = V[0:K1, :]
    Z1 = (U1 * S1).dot(V1)
    ax.pcolor(Z1)
    ax.set_title('image (z1)')
    ax = axs[1, 0]
    K2 = 2
    U2 = U[:, 0:K2]
    S2 = S[0:K2, 0:K2]
    V2 = V[0:K2, :]
    Z2 = (U2.dot(S2)).dot(V2)
    ax.pcolor(Z2)
    ax.set_title('image (z2)')
    ax = axs[1, 1]
    K3 = 5
    U3 = U[:, 0:K3]
    S3 = S[0:K3, 0:K3]
    V3 = V[0:K3, :]
    Z3 = (U3.dot(S3)).dot(V3)
    ax.pcolor(Z3)
    ax.set_title('image (z3)')

    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.show()
# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    # firstMatrix()
    # firstSVDcal()
    # PODclass()
    # scalaPOD()
    vectorPOD()
# See PyCharm he
# lp at https://www.jetbrains.com/help/pycharm/
