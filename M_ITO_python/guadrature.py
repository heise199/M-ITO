"""
高斯积分模块 - 生成高斯积分点和权重
"""
import numpy as np


def guadrature(quadorder, dim):
    """
    生成高斯积分点和权重
    
    参数:
        quadorder: 积分阶数 (3点高斯积分)
        dim: 维度 (2D)
    
    返回:
        quadweight: 积分权重
        quadpoint: 积分点坐标
    """
    quadpoint = np.zeros((quadorder**dim, dim))
    quadweight = np.zeros(quadorder**dim)
    
    # 3点高斯积分的节点和权重
    r1pt = np.zeros(quadorder)
    r1wt = np.zeros(quadorder)
    r1pt[0] = 0.774596669241483
    r1pt[1] = -0.774596669241483
    r1pt[2] = 0.000000000000000
    r1wt[0] = 0.555555555555556
    r1wt[1] = 0.555555555555556
    r1wt[2] = 0.888888888888889
    
    n = 0
    for i in range(quadorder):
        for j in range(quadorder):
            quadpoint[n, :] = [r1pt[i], r1pt[j]]
            quadweight[n] = r1wt[i] * r1wt[j]
            n += 1
    
    return quadweight, quadpoint

