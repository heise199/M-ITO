"""
高斯积分 (Gauss Quadrature)
从 MATLAB 代码转换
"""

import numpy as np


def guadrature(quadorder, dim):
    """
    生成高斯积分点和权重
    
    参数:
        quadorder: 积分阶数 (每个维度的点数)
        dim: 维度
    
    返回:
        quadweight: 积分权重 (quadorder^dim,)
        quadpoint: 积分点坐标 (quadorder^dim, dim)
    
    改编自 Gauss-Legendre 积分
    """
    # 3点高斯积分的点和权重
    r1pt = np.array([
        0.774596669241483,
        -0.774596669241483,
        0.000000000000000
    ])
    
    r1wt = np.array([
        0.555555555555556,
        0.555555555555556,
        0.888888888888889
    ])
    
    # 生成2D积分点和权重
    quadpoint = np.zeros((quadorder**dim, dim))
    quadweight = np.zeros(quadorder**dim)
    
    n = 0
    for i in range(quadorder):
        for j in range(quadorder):
            quadpoint[n, :] = [r1pt[i], r1pt[j]]
            quadweight[n] = r1wt[i] * r1wt[j]
            n += 1
    
    return quadweight, quadpoint


# ======================================================================================================================
# 函数: guadrature
#
# 用于等几何拓扑优化的紧凑高效 Python 实现
#
# 开发者: 原始 MATLAB 代码 - Jie Gao
# Email: JieGao@hust.edu.cn
# Python 转换: 2025
#
# 主要参考文献:
#
# (1) Jie Gao, Lin Wang, Zhen Luo, Liang Gao. IgaTop: an implementation of topology optimization for structures
# using IGA in Matlab. Structural and multidisciplinary optimization.
#
# (2) Jie Gao, Liang Gao, Zhen Luo, Peigen Li. Isogeometric topology optimization for continuum structures using
# density distribution function. Int J Numer Methods Eng, 2019, 119:991–1017
#
# *********************************************   免责声明   *******************************************************
# 作者保留程序的所有权利。程序可用于学术和教育目的。作者不保证代码没有错误，
# 并且不对因使用程序而引起的任何事件承担责任。
# ======================================================================================================================

