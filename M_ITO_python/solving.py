"""
求解器 (Solver) - 有限元方程求解
从 MATLAB 代码转换
"""

import numpy as np
from scipy.sparse.linalg import spsolve


def solving(CtrPts, DBoudary, Dofs, K, F, BoundCon):
    """
    求解有限元方程 KU = F
    
    参数:
        CtrPts: 控制点信息字典
                - Num: 控制点总数
        DBoudary: Dirichlet 边界条件字典
        Dofs: 自由度信息字典
              - Num: 总自由度数
        K: 刚度矩阵
        F: 载荷向量
        BoundCon: 边界条件类型 (1-5)
    
    返回:
        U: 位移向量
    
    改编自 MATLAB IgaTop2D 代码
    """
    if BoundCon in [1, 4, 5]:
        U_fixeddofs = DBoudary['CtrPtsOrd']
        V_fixeddofs = DBoudary['CtrPtsOrd'] + CtrPts['Num']
    elif BoundCon in [2, 3]:
        # 确保是数组
        U_fixeddofs = np.atleast_1d(DBoudary['CtrPtsOrd1'])
        V_fixeddofs = np.concatenate([
            np.atleast_1d(DBoudary['CtrPtsOrd1']),
            np.atleast_1d(DBoudary['CtrPtsOrd2'])
        ]) + CtrPts['Num']
    
    # 转换为0-based索引
    U_fixeddofs_0 = U_fixeddofs - 1
    V_fixeddofs_0 = V_fixeddofs - 1
    
    Dofs['Ufixed'] = U_fixeddofs
    Dofs['Vfixed'] = V_fixeddofs
    
    # 组合固定自由度
    fixed_dofs = np.concatenate([U_fixeddofs_0, V_fixeddofs_0])
    
    # 自由自由度 (0-based)
    all_dofs = np.arange(Dofs['Num'])
    Dofs['Free'] = np.setdiff1d(all_dofs, fixed_dofs)
    
    # 初始化位移向量
    U = np.zeros((Dofs['Num'], 1))
    
    # 求解线性方程组
    # 模拟 MATLAB 的 \ 运算符行为：对奇异或接近奇异的矩阵使用最小二乘求解
    K_free = K[np.ix_(Dofs['Free'], Dofs['Free'])]
    F_free = F[Dofs['Free']].flatten()
    
    # 检查矩阵对角线是否有零元素（奇异矩阵的指标）
    diag = K_free.diagonal()
    zero_diag_count = np.sum(np.abs(diag) < 1e-12)
    
    if zero_diag_count > 0:
        # 矩阵奇异或接近奇异，使用 LSQR 求解（类似 MATLAB 的 \ 行为）
        from scipy.sparse.linalg import lsqr
        U_free, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = lsqr(K_free, F_free, atol=1e-12, btol=1e-12, iter_lim=10000)
        U_free = U_free.reshape(-1, 1)
    else:
        # 矩阵满秩，使用标准的稀疏直接求解器
        try:
            U_free = spsolve(K_free, F_free).reshape(-1, 1)
            # 检查解的有效性
            if np.any(np.isnan(U_free)) or np.any(np.isinf(U_free)):
                from scipy.sparse.linalg import lsqr
                U_free, istop, itn, r1norm = lsqr(K_free, F_free, atol=1e-12, btol=1e-12)[:4]
                U_free = U_free.reshape(-1, 1)
        except Exception as e:
            # 直接求解失败，使用 LSQR
            from scipy.sparse.linalg import lsqr
            U_free, istop, itn, r1norm = lsqr(K_free, F_free, atol=1e-12, btol=1e-12)[:4]
            U_free = U_free.reshape(-1, 1)
    
    U[Dofs['Free']] = U_free
    
    return U


# ======================================================================================================================
# 函数: solving
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

