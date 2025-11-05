"""
2D 刚度矩阵组装 (2D Stiffness Matrix Assembly)
从 MATLAB 代码转换
"""

import numpy as np
from scipy.sparse import coo_matrix


def stiff_ass2d(KE, CtrPts, Ele, Dim, Dofs_Num):
    """
    组装全局刚度矩阵
    
    参数:
        KE: 单元刚度矩阵列表
        CtrPts: 控制点信息字典
                - Num: 控制点总数
        Ele: 单元信息字典
             - Num: 单元总数
             - CtrPtsCon: 单元控制点连接
             - CtrPtsNum: 每个单元的控制点数
        Dim: 维度
        Dofs_Num: 总自由度数
    
    返回:
        K: 全局刚度矩阵 (稀疏)
    
    改编自 MATLAB IgaTop2D 代码
    """
    # 预分配数组
    total_entries = Ele['Num'] * Dim * Ele['CtrPtsNum'] * Dim * Ele['CtrPtsNum']
    II = np.zeros(total_entries)
    JJ = np.zeros(total_entries)
    KX = np.zeros(total_entries)
    
    ntriplets = 0
    for ide in range(Ele['Num']):
        Ele_NoCtPt = Ele['CtrPtsCon'][ide, :]  # 单元控制点编号 (1-based)
        
        # 构建自由度索引 (转为 0-based)
        edof = np.concatenate([Ele_NoCtPt - 1, Ele_NoCtPt - 1 + CtrPts['Num']])
        
        # 填充三元组
        for krow in range(len(edof)):
            for kcol in range(len(edof)):
                II[ntriplets] = edof[krow]
                JJ[ntriplets] = edof[kcol]
                KX[ntriplets] = KE[ide][krow, kcol]
                ntriplets += 1
    
    # 创建稀疏矩阵
    K = coo_matrix((KX[:ntriplets], (II[:ntriplets].astype(int), JJ[:ntriplets].astype(int))), 
                   shape=(Dofs_Num, Dofs_Num)).tocsr()
    
    # 对称化
    K = (K + K.T) / 2
    
    return K


# ======================================================================================================================
# 函数: stiff_ass2d
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

