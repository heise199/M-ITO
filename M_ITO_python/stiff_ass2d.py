"""
刚度矩阵组装模块 - 组装全局刚度矩阵
"""
import numpy as np
from scipy.sparse import csr_matrix


def stiff_ass2d(KE, CtrPts, Ele, Dim, Dofs_Num):
    """
    组装全局刚度矩阵
    
    参数:
        KE: 单元刚度矩阵列表
        CtrPts: 控制点信息
        Ele: 单元信息
        Dim: 维度
        Dofs_Num: 自由度总数
    
    返回:
        K: 全局刚度矩阵（稀疏矩阵）
    """
    II = []
    JJ = []
    KX = []
    
    for ide in range(Ele['Num']):
        Ele_NoCtPt = Ele['CtrPtsCon'][ide, :] - 1  # 转换为0-based索引
        edof = np.concatenate([Ele_NoCtPt, Ele_NoCtPt + CtrPts['Num']])
        
        for krow in range(len(edof)):
            for kcol in range(len(edof)):
                II.append(int(edof[krow]))
                JJ.append(int(edof[kcol]))
                KX.append(KE[ide][krow, kcol])
    
    # 创建稀疏矩阵
    K = csr_matrix((KX, (II, JJ)), shape=(Dofs_Num, Dofs_Num))
    
    # 对称化
    K = (K + K.T) / 2
    
    return K

