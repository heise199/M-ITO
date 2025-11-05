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
    if BoundCon == 1:  # 悬臂梁
        DBoudary['CtrPtsOrd'] = CtrPts['Seque'][:, 0]  # 第一列固定
        load_u = 1.0
        load_v = 0.5
        N, id_vals = nrbbasisfun(np.array([[load_u], [load_v]]), NURBS)
        NBoudary_CtrPtsOrd = id_vals.flatten()
        NBoudary_N = N.flatten()
    
    elif BoundCon == 2:  # MBB 梁
        DBoudary['CtrPtsOrd1'] = CtrPts['Seque'][0, 0]  # 左下角
        DBoudary['CtrPtsOrd2'] = CtrPts['Seque'][0, -1]  # 右下角
        load_u = 0.5
        load_v = 1.0
        N, id_vals = nrbbasisfun(np.array([[load_u], [load_v]]), NURBS)
        NBoudary_CtrPtsOrd = id_vals.flatten()
        NBoudary_N = N.flatten()
    
    elif BoundCon == 3:  # Michell 型结构
        DBoudary['CtrPtsOrd1'] = CtrPts['Seque'][0, 0]  # 左下角
        DBoudary['CtrPtsOrd2'] = CtrPts['Seque'][0, -1]  # 右下角
        load_u = 0.5
        load_v = 0.0
        N, id_vals = nrbbasisfun(np.array([[load_u], [load_v]]), NURBS)
        NBoudary_CtrPtsOrd = id_vals.flatten()
        NBoudary_N = N.flatten()
    
    elif BoundCon == 4:  # L 梁
        DBoudary['CtrPtsOrd'] = CtrPts['Seque'][:, 0]  # 第一列固定
        load_u = 1.0
        load_v = 1.0
        N, id_vals = nrbbasisfun(np.array([[load_u], [load_v]]), NURBS)
        NBoudary_CtrPtsOrd = id_vals.flatten()
        NBoudary_N = N.flatten()
    
    elif BoundCon == 5:  # 四分之一环形
        DBoudary['CtrPtsOrd'] = CtrPts['Seque'][:, -1]  # 最后一列固定
        load_u = 0.0
        load_v = 1.0
        N, id_vals = nrbbasisfun(np.array([[load_u], [load_v]]), NURBS)
        NBoudary_CtrPtsOrd = id_vals.flatten()
        NBoudary_N = N.flatten()
    
    # 初始化载荷向量
    F = np.zeros((Dofs_Num, 1))
    
    # 施加载荷
    # 注意：MATLAB 对重复索引赋值会累加，NumPy 会覆盖，必须使用 np.add.at()
    if BoundCon in [1, 2, 3, 4]:
        # 在 Y 方向施加载荷
        load_indices = (NBoudary_CtrPtsOrd - 1 + CtrPts['Num']).astype(int)
        load_values = (-1.0 * NBoudary_N).reshape(-1, 1)
        # 使用 np.add.at 来累加重复索引的载荷（MATLAB 行为）
        np.add.at(F, load_indices, load_values)
        
        # 临时调试：检查载荷（已禁用）
        if False and np.count_nonzero(F) == 0:
            print(f'\n[警告] 载荷为零！BoundCon={BoundCon}')
            print(f'  load_u={load_u}, load_v={load_v}')
            print(f'  N非零数量: {np.count_nonzero(N)}, N.shape: {N.shape}')
            print(f'  N值: {N.flatten()[:9]}')
            print(f'  id_vals: {id_vals.flatten()[:9]}')
            print(f'  NURBS.knots[0] 范围: [{NURBS["knots"][0][0]}, {NURBS["knots"][0][-1]}]')
            print(f'  NURBS.knots[1] 范围: [{NURBS["knots"][1][0]}, {NURBS["knots"][1][-1]}]')
            print(f'  NURBS.number: {NURBS["number"]}')
    
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

