"""
边界条件 (Boundary Conditions)
从 MATLAB 代码转换
"""

import numpy as np
from nurbs import nrbbasisfun


def boun_cond(CtrPts, BoundCon, NURBS, Dofs_Num):
    """
    设置边界条件
    
    参数:
        CtrPts: 控制点信息字典
                - Seque: 控制点序列矩阵
                - Num: 控制点总数
        BoundCon: 边界条件类型 (1-5)
        NURBS: NURBS 几何结构
        Dofs_Num: 总自由度数
    
    返回:
        DBoudary: Dirichlet 边界条件字典
        F: 载荷向量
    
    改编自 MATLAB IgaTop2D 代码
    """
    DBoudary = {}
    
    # 根据边界条件类型设置 Dirichlet 边界条件和载荷
    if BoundCon == 1:  # Cantilever beam
        DBoudary['CtrPtsOrd'] = CtrPts['Seque'][:, 0]  # Fix left edge
        # Use exact boundary value to match MATLAB behavior
        load_u = 1.0
        load_v = 0.5
        N, id_vals = nrbbasisfun(np.array([[load_u], [load_v]]), NURBS)
        
        # Filter out basis functions with negligible weights
        N_flat = N.flatten()
        id_flat = id_vals.flatten()
        significant_mask = np.abs(N_flat) >= 1e-10  # Keep only truly non-zero weights
        NBoudary_CtrPtsOrd = id_flat[significant_mask]
        NBoudary_N = N_flat[significant_mask]
        # Renormalize to ensure sum equals 1.0 exactly
        if len(NBoudary_N) > 0:
            NBoudary_N = NBoudary_N / np.sum(NBoudary_N)
    
    elif BoundCon == 2:  # MBB beam
        DBoudary['CtrPtsOrd1'] = CtrPts['Seque'][0, 0]  # Bottom-left corner
        DBoudary['CtrPtsOrd2'] = CtrPts['Seque'][0, -1]  # Bottom-right corner
        load_u = 0.5
        load_v = 1.0 - 1e-6  # Near top boundary
        N, id_vals = nrbbasisfun(np.array([[load_u], [load_v]]), NURBS)
        # Filter out basis functions with negligible weights
        N_flat = N.flatten()
        id_flat = id_vals.flatten()
        significant_mask = np.abs(N_flat) >= 1e-4
        NBoudary_CtrPtsOrd = id_flat[significant_mask]
        NBoudary_N = N_flat[significant_mask]
        # Renormalize to ensure sum equals 1.0 exactly
        NBoudary_N = NBoudary_N / np.sum(NBoudary_N)
    
    elif BoundCon == 3:  # Michell-type structure
        DBoudary['CtrPtsOrd1'] = CtrPts['Seque'][0, 0]  # Bottom-left corner
        DBoudary['CtrPtsOrd2'] = CtrPts['Seque'][0, -1]  # Bottom-right corner
        load_u = 0.5
        load_v = 1e-6  # Near bottom boundary
        N, id_vals = nrbbasisfun(np.array([[load_u], [load_v]]), NURBS)
        # Filter out basis functions with negligible weights
        N_flat = N.flatten()
        id_flat = id_vals.flatten()
        significant_mask = np.abs(N_flat) >= 1e-10  # Keep only truly non-zero weights
        NBoudary_CtrPtsOrd = id_flat[significant_mask]
        NBoudary_N = N_flat[significant_mask]
        # Renormalize to ensure sum equals 1.0 exactly
        if len(NBoudary_N) > 0:
            NBoudary_N = NBoudary_N / np.sum(NBoudary_N)
    
    elif BoundCon == 4:  # L beam
        DBoudary['CtrPtsOrd'] = CtrPts['Seque'][:, 0]  # Fix left edge
        load_u = 1.0 - 1e-6  # Near right boundary
        load_v = 1.0 - 1e-6  # Near top boundary
        N, id_vals = nrbbasisfun(np.array([[load_u], [load_v]]), NURBS)
        # Filter out basis functions with negligible weights
        N_flat = N.flatten()
        id_flat = id_vals.flatten()
        significant_mask = np.abs(N_flat) >= 1e-4
        NBoudary_CtrPtsOrd = id_flat[significant_mask]
        NBoudary_N = N_flat[significant_mask]
        # Renormalize to ensure sum equals 1.0 exactly
        NBoudary_N = NBoudary_N / np.sum(NBoudary_N)
    
    elif BoundCon == 5:  # A quarter annulus
        DBoudary['CtrPtsOrd'] = CtrPts['Seque'][:, -1]  # Fix right edge
        load_u = 1e-6  # Near left boundary
        load_v = 1.0 - 1e-6  # Near top boundary
        N, id_vals = nrbbasisfun(np.array([[load_u], [load_v]]), NURBS)
        # Filter out basis functions with negligible weights
        N_flat = N.flatten()
        id_flat = id_vals.flatten()
        significant_mask = np.abs(N_flat) >= 1e-4
        NBoudary_CtrPtsOrd = id_flat[significant_mask]
        NBoudary_N = N_flat[significant_mask]
        # Renormalize to ensure sum equals 1.0 exactly
        NBoudary_N = NBoudary_N / np.sum(NBoudary_N)
    
    # Initialize force vector
    F = np.zeros((Dofs_Num, 1))
    
    # Apply loads
    # Note: MATLAB accumulates for duplicate indices, NumPy overwrites, must use np.add.at()
    if BoundCon in [1, 2, 3, 4]:
        # Apply load in Y direction
        load_indices = (NBoudary_CtrPtsOrd - 1 + CtrPts['Num']).astype(int)
        load_values = (-1.0 * NBoudary_N).reshape(-1, 1)
        # 使用 np.add.at 来累加重复索引的载荷（MATLAB 行为）
        np.add.at(F, load_indices, load_values)
    
    elif BoundCon == 5:
        # 在 X 方向施加载荷
        load_indices = (NBoudary_CtrPtsOrd - 1).astype(int)
        load_values = (1.0 * NBoudary_N).reshape(-1, 1)  # 正方向
        # 使用 np.add.at 来累加重复索引的载荷（MATLAB 行为）
        np.add.at(F, load_indices, load_values)
    
    return DBoudary, F


# ======================================================================================================================
# 函数: boun_cond
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

