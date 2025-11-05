"""
Shepard 函数 (Shepard Function) - 平滑机制
从 MATLAB 代码转换
"""

import numpy as np
from scipy.sparse import coo_matrix


def shep_fun(CtrPts, rmin):
    """
    计算 Shepard 函数用于平滑设计变量
    
    参数:
        CtrPts: 控制点字典
                - NumU: U方向控制点数量
                - NumV: V方向控制点数量
        rmin: 最小半径
    
    返回:
        Sh: Shepard 函数稀疏矩阵
        Hs: Sh 行和向量
    
    改编自 MATLAB IgaTop2D 代码
    """
    Ctr_NumU = CtrPts['NumU']
    Ctr_NumV = CtrPts['NumV']
    
    # 预分配数组
    total_size = Ctr_NumU * Ctr_NumV * (2 * (int(np.ceil(rmin)) - 1) + 1) ** 2
    iH = np.ones(total_size)
    jH = np.ones(total_size)
    sH = np.zeros(total_size)
    
    k = 0
    for j1 in range(1, Ctr_NumV + 1):
        for i1 in range(1, Ctr_NumU + 1):
            e1 = (j1 - 1) * Ctr_NumU + i1
            
            j2_start = max(j1 - (int(np.ceil(rmin)) - 1), 1)
            j2_end = min(j1 + (int(np.ceil(rmin)) - 1), Ctr_NumV)
            
            for j2 in range(j2_start, j2_end + 1):
                i2_start = max(i1 - (int(np.ceil(rmin)) - 1), 1)
                i2_end = min(i1 + (int(np.ceil(rmin)) - 1), Ctr_NumU)
                
                for i2 in range(i2_start, i2_end + 1):
                    e2 = (j2 - 1) * Ctr_NumU + i2
                    
                    iH[k] = e1
                    jH[k] = e2
                    
                    theta = np.sqrt((j1 - j2)**2 + (i1 - i2)**2) / rmin / np.sqrt(2)
                    sH[k] = (max(0, (1 - theta))**6) * (35 * theta**2 + 18 * theta + 3)
                    k += 1
    
    # 截断到实际使用的大小
    iH = iH[:k]
    jH = jH[:k]
    sH = sH[:k]
    
    # 创建稀疏矩阵 (转换为 0-based 索引)
    Sh = coo_matrix((sH, (iH.astype(int) - 1, jH.astype(int) - 1))).tocsr()
    
    # 计算行和
    Hs = np.array(Sh.sum(axis=1)).flatten()
    
    return Sh, Hs


# ======================================================================================================================
# 函数: shep_fun
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

