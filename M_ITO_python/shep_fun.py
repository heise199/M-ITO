"""
形状函数模块 - 计算Shepard函数用于密度过滤
"""
import numpy as np
from scipy.sparse import csr_matrix


def shep_fun(CtrPts, rmin):
    """
    计算Shepard函数用于密度过滤
    
    参数:
        CtrPts: 控制点信息
        rmin: 最小过滤半径
    
    返回:
        Sh: 过滤矩阵
        Hs: 行和向量
    """
    Ctr_NumU = CtrPts['NumU']
    Ctr_NumV = CtrPts['NumV']
    
    # 预分配稀疏矩阵的索引和值
    max_neighbors = (2 * (int(np.ceil(rmin)) - 1) + 1) ** 2
    iH = []
    jH = []
    sH = []
    
    for j1 in range(Ctr_NumV):
        for i1 in range(Ctr_NumU):
            e1 = j1 * Ctr_NumU + i1 + 1  # MATLAB索引从1开始
            
            # 计算邻居范围
            j2_min = max(j1 - (int(np.ceil(rmin)) - 1), 0)
            j2_max = min(j1 + (int(np.ceil(rmin)) - 1), Ctr_NumV - 1)
            i2_min = max(i1 - (int(np.ceil(rmin)) - 1), 0)
            i2_max = min(i1 + (int(np.ceil(rmin)) - 1), Ctr_NumU - 1)
            
            for j2 in range(j2_min, j2_max + 1):
                for i2 in range(i2_min, i2_max + 1):
                    e2 = j2 * Ctr_NumU + i2 + 1  # MATLAB索引从1开始
                    
                    theta = np.sqrt((j1 - j2)**2 + (i1 - i2)**2) / rmin / np.sqrt(2)
                    weight = (max(0, 1 - theta) ** 6) * (35 * theta**2 + 18 * theta + 3)
                    
                    iH.append(e1 - 1)  # 转换为0-based索引
                    jH.append(e2 - 1)  # 转换为0-based索引
                    sH.append(weight)
    
    # 创建稀疏矩阵
    Sh = csr_matrix((sH, (iH, jH)), shape=(Ctr_NumU * Ctr_NumV, Ctr_NumU * Ctr_NumV))
    Hs = np.array(Sh.sum(axis=1)).flatten()
    
    return Sh, Hs

