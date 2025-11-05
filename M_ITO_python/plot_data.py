"""
绘图数据准备模块 - 准备可视化所需的数据
"""
import numpy as np
from nrbbasisfun import nrbbasisfun
from scipy.sparse import csr_matrix


def plot_data(Num, NURBS):
    """
    准备绘图数据
    
    参数:
        Num: 网格数量 [n, m]
        NURBS: NURBS曲面对象
    
    返回:
        DenFied: 密度场数据字典
        Pos: 图形位置信息字典
    """
    # 图形位置（简化版，可根据需要调整）
    Pos = {}
    Pos['p1'] = [100, 400, 600, 400]
    Pos['p2'] = [700, 400, 600, 400]
    Pos['p3'] = [100, 50, 600, 300]
    Pos['p4'] = [700, 50, 600, 300]
    
    # 生成用于绘图的参数网格
    Uknots = np.linspace(0, 1, 10 * Num[0])
    Vknots = np.linspace(0, 1, 10 * Num[1])
    
    # 计算基函数与索引（严格复刻MATLAB：一次性传入 cell {Uknots, Vknots}，u 方向变化最快）
    uv_cell = {0: Uknots, 1: Vknots}
    N_f, id_f = nrbbasisfun(uv_cell, NURBS)
    
    # 计算物理坐标
    PCor_U = []
    PCor_W = []
    
    for v_val in Vknots:
        for u_val in Uknots:
            pt = NURBS.evaluate_single((u_val, v_val))
            PCor_U.append([pt[0] / pt[3], pt[1] / pt[3], pt[2] / pt[3]])
            PCor_W.append(pt[3])
    
    # 这里的PCor_U已是物理坐标（已除以w），不要再次除以权重
    PCor_U = np.array(PCor_U).T  # shape (3, n_points)
    PCor_W = np.array(PCor_W)
    
    PCor_Ux = PCor_U[0, :].reshape(len(Vknots), len(Uknots))
    PCor_Uy = PCor_U[1, :].reshape(len(Vknots), len(Uknots))
    
    DenFied = {}
    DenFied['N'] = N_f
    DenFied['id'] = id_f
    DenFied['U'] = Uknots
    DenFied['V'] = Vknots
    DenFied['Ux'] = PCor_Ux
    DenFied['Uy'] = PCor_Uy
    # 构建稀疏映射矩阵：将控制点密度映射到绘图网格 DDF
    rows = []
    cols = []
    data = []
    npts = N_f.shape[0]
    for i in range(npts):
        for j in range(N_f.shape[1]):
            rows.append(i)
            cols.append(int(id_f[i, j]) - 1)
            data.append(N_f[i, j])
    DenFied['R'] = csr_matrix((data, (rows, cols)), shape=(npts, NURBS.ctrlpts_size_u * NURBS.ctrlpts_size_v))
    
    return DenFied, Pos

