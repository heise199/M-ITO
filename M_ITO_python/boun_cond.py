"""
边界条件模块 - 设置边界条件和载荷
"""
import numpy as np
from geomdl import utilities


def boun_cond(CtrPts, BoundCon, NURBS, Dofs_Num):
    """
    设置边界条件和载荷
    
    参数:
        CtrPts: 控制点信息
        BoundCon: 边界条件类型
        NURBS: NURBS曲面对象
        Dofs_Num: 自由度总数
    
    返回:
        DBoudary: 边界条件信息字典
        F: 载荷向量
    """
    DBoudary = {}
    
    if BoundCon == 1:  # Cantilever beam
        DBoudary['CtrPtsOrd'] = CtrPts['Seque'][:, 0]  # 左边界
        load_u = 1.0
        load_v = 0.5
        
    elif BoundCon == 2:  # MBB beam
        DBoudary['CtrPtsOrd1'] = CtrPts['Seque'][0, 0]
        DBoudary['CtrPtsOrd2'] = CtrPts['Seque'][0, -1]
        load_u = 0.5
        load_v = 1.0
        
    elif BoundCon == 3:  # Michell-type structure
        DBoudary['CtrPtsOrd1'] = CtrPts['Seque'][0, 0]
        DBoudary['CtrPtsOrd2'] = CtrPts['Seque'][0, -1]
        load_u = 0.5
        load_v = 0.0
        
    elif BoundCon == 4:  # L beam
        DBoudary['CtrPtsOrd'] = CtrPts['Seque'][:, 0]
        load_u = 1.0
        load_v = 1.0
        
    elif BoundCon == 5:  # A quarter annulus
        DBoudary['CtrPtsOrd'] = CtrPts['Seque'][:, -1]
        load_u = 0.0
        load_v = 1.0
    
    # 计算载荷点的基函数值
    # 找到参数值对应的基函数
    from nrbbasisfun import nrbbasisfun
    uv_load = np.array([[load_u], [load_v]])  # shape (2, 1)
    N, id_array = nrbbasisfun(uv_load, NURBS)
    
    NBoudary = {}
    # MATLAB: NBoudary.CtrPtsOrd = id'; id是(1, n_basis)，转置后是(n_basis, 1)，然后flatten
    NBoudary['CtrPtsOrd'] = id_array.flatten()  # MATLAB索引从1开始
    NBoudary['N'] = N.flatten()
    
    # 调试信息（仅在需要时打印）
    if np.sum(np.abs(NBoudary['N'])) < 1e-10:
        print(f"警告: 载荷点 ({load_u}, {load_v}) 的基函数值全为0")
        print(f"  id_array: {id_array}, N: {N}")
    
    # 创建载荷向量
    F = np.zeros(Dofs_Num)
    
    if BoundCon in [1, 2, 3, 4]:
        # Y方向载荷
        # MATLAB: F(NBoudary.CtrPtsOrd+CtrPts.Num) = -1*NBoudary.N;
        # MATLAB索引从1开始，Python从0开始
        for i, ctrlpt_idx in enumerate(NBoudary['CtrPtsOrd']):
            # ctrlpt_idx是MATLAB索引（从1开始），转换为Python索引（从0开始）
            # 然后加上CtrPts.Num得到y方向自由度
            dof_idx = int(ctrlpt_idx) - 1 + CtrPts['Num']  # Python索引，y方向自由度
            if 0 <= dof_idx < Dofs_Num:
                F[dof_idx] = -1.0 * NBoudary['N'][i]  # MATLAB直接赋值，不是累加
    
    elif BoundCon == 5:
        # X方向载荷
        # MATLAB: F(NBoudary.CtrPtsOrd) = -1*NBoudary.N;
        for i, ctrlpt_idx in enumerate(NBoudary['CtrPtsOrd']):
            dof_idx = int(ctrlpt_idx) - 1  # Python索引，x方向自由度
            if 0 <= dof_idx < Dofs_Num:
                F[dof_idx] = -1.0 * NBoudary['N'][i]  # MATLAB直接赋值
    
    return DBoudary, F

