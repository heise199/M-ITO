"""
几何模型 (Geometry Model)
从 MATLAB 代码转换
"""

import numpy as np
from nurbs import nrbmak, nrbdegelev, nrbkntins


def geom_mod(L, W, Order, Num, BoundCon):
    """
    创建几何模型 NURBS 结构
    
    参数:
        L: 长度
        W: 宽度
        Order: 提升次数 [Order_U, Order_V]
        Num: 插入节点后的数量 [Num_U, Num_V]
        BoundCon: 边界条件类型 (1-5)
    
    返回:
        NURBS: NURBS 几何结构
    
    改编自 MATLAB IgaTop2D 代码
    """
    if BoundCon in [1, 2, 3]:
        # 矩形域
        knots = [[0, 0, 1, 1], [0, 0, 1, 1]]
        ControlPts = np.zeros((4, 2, 2))
        ControlPts[:, :, 0] = np.array([[0, L], [0, 0], [0, 0], [1, 1]])
        ControlPts[:, :, 1] = np.array([[0, L], [W, W], [0, 0], [1, 1]])
    
    elif BoundCon == 4:
        # L 形梁
        knots = [[0, 0, 0.5, 1, 1], [0, 0, 1, 1]]
        ControlPts = np.zeros((4, 3, 2))
        ControlPts[:, :, 0] = np.array([[0, 0, L], [L, 0, 0], [0, 0, 0], [1, 1, 1]])
        ControlPts[:, :, 1] = np.array([[W, W, L], [L, W, W], [0, 0, 0], [1, 1, 1]])
    
    elif BoundCon == 5:
        # 四分之一环形
        W = W / 2
        knots = [[0, 0, 0, 1, 1, 1], [0, 0, 1, 1]]
        ControlPts = np.zeros((4, 3, 2))
        ControlPts[:, :, 0] = np.array([[0, W, W], [W, W, 0], [0, 0, 0], [1, np.sqrt(2)/2, 1]])
        ControlPts[:, :, 1] = np.array([[0, L, L], [L, L, 0], [0, 0, 0], [1, np.sqrt(2)/2, 1]])
    
    # 将控制点转换为齐次坐标
    coefs = np.zeros(ControlPts.shape)
    coefs[0, :, :] = ControlPts[0, :, :] * ControlPts[3, :, :]
    coefs[1, :, :] = ControlPts[1, :, :] * ControlPts[3, :, :]
    coefs[2, :, :] = ControlPts[2, :, :] * ControlPts[3, :, :]
    coefs[3, :, :] = ControlPts[3, :, :]
    
    # 创建 NURBS
    NURBS = nrbmak(coefs, knots)
    
    # 提升次数
    NURBS = nrbdegelev(NURBS, Order)
    
    # 插入节点
    iknot_u = np.linspace(0, 1, Num[0])
    iknot_v = np.linspace(0, 1, Num[1])
    
    # 找出不在现有节点序列中的节点
    knots_to_insert_u = np.setdiff1d(iknot_u, NURBS['knots'][0])
    knots_to_insert_v = np.setdiff1d(iknot_v, NURBS['knots'][1])
    
    NURBS = nrbkntins(NURBS, [knots_to_insert_u, knots_to_insert_v])
    
    return NURBS


# ======================================================================================================================
# 函数: geom_mod
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

