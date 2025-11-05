"""
求解模块 - 求解线性方程组
"""
import numpy as np
from scipy.sparse.linalg import spsolve


def solving(CtrPts, DBoudary, Dofs, K, F, BoundCon):
    """
    求解线性方程组
    
    参数:
        CtrPts: 控制点信息
        DBoudary: 边界条件信息
        Dofs: 自由度信息
        K: 全局刚度矩阵
        F: 载荷向量
        BoundCon: 边界条件类型
    
    返回:
        U: 位移向量
    """
    # MATLAB代码中索引从1开始，Python需要转换为0-based
    # MATLAB: U_fixeddofs = DBoudary.CtrPtsOrd; (MATLAB索引，1-based)
    # MATLAB: V_fixeddofs = DBoudary.CtrPtsOrd + CtrPts.Num; (MATLAB索引，1-based)
    
    if BoundCon in [1, 4, 5]:
        # MATLAB索引转换为Python索引（减1，因为MATLAB从1开始）
        U_fixeddofs = np.atleast_1d(DBoudary['CtrPtsOrd'] - 1)  # MATLAB索引转Python索引（0-based）
        V_fixeddofs = np.atleast_1d(DBoudary['CtrPtsOrd'] - 1 + CtrPts['Num'])  # y方向自由度（0-based）
    elif BoundCon in [2, 3]:
        U_fixeddofs = np.atleast_1d(DBoudary['CtrPtsOrd1'] - 1)  # MATLAB索引转Python索引（0-based）
        V_fixeddofs = np.array([
            DBoudary['CtrPtsOrd1'] - 1 + CtrPts['Num'],
            DBoudary['CtrPtsOrd2'] - 1 + CtrPts['Num']
        ])
    
    Dofs['Ufixed'] = U_fixeddofs
    Dofs['Vfixed'] = V_fixeddofs
    
    # 确保所有数组都是1维的，然后连接
    U_fixed = np.atleast_1d(Dofs['Ufixed'])
    V_fixed = np.atleast_1d(Dofs['Vfixed'])
    all_fixed = np.concatenate([U_fixed, V_fixed])
    
    # MATLAB: Dofs.Free = setdiff(1:Dofs.Num,[Dofs.Ufixed; Dofs.Vfixed]);
    # MATLAB中1:Dofs.Num是1-based索引[1,2,...,Dofs.Num]
    # 转换为Python的0-based: [0,1,...,Dofs.Num-1]
    # all_fixed已经是0-based，所以直接使用
    Dofs['Free'] = np.setdiff1d(np.arange(Dofs['Num']), all_fixed)
    
    U = np.zeros(Dofs['Num'])
    
    # 检查刚度矩阵的条件
    K_free = K[Dofs['Free'], :][:, Dofs['Free']]
    if K_free.shape[0] != len(Dofs['Free']):
        print(f"警告: 刚度矩阵维度不匹配: K_free.shape={K_free.shape}, Free长度={len(Dofs['Free'])}")
    
    # 求解自由度的位移
    try:
        U[Dofs['Free']] = spsolve(K_free, F[Dofs['Free']])
    except Exception as e:
        print(f"求解失败: {e}")
        print(f"K_free形状: {K_free.shape}, 秩: {np.linalg.matrix_rank(K_free.toarray()) if hasattr(K_free, 'toarray') else 'N/A'}")
        print(f"Free自由度数量: {len(Dofs['Free'])}, Fixed自由度数量: {len(all_fixed)}")
        raise
    
    return U

