"""
主程序 - IGA拓扑优化2D
"""
import numpy as np
from geom_mod import geom_mod
from pre_iga import pre_iga
from boun_cond import boun_cond
from shep_fun import shep_fun
from guadrature import guadrature
from nrbbasisfun import nrbbasisfun
from nrbbasisfunder import nrbbasisfunder
from stiff_ele2d import stiff_ele2d
from stiff_ass2d import stiff_ass2d
from solving import solving
from oc import oc
from plot_data import plot_data
from plot_topy import plot_topy
from scipy.sparse import csr_matrix


def iga_top2d(L, W, Order, Num, BoundCon, Vmax, penal, rmin):
    """
    IGA拓扑优化主程序
    
    参数:
        L: 长度
        W: 宽度
        Order: B样条阶数 [p, q]
        Num: 控制点数量 [n, m]
        BoundCon: 边界条件类型 (1-5)
        Vmax: 最大体积分数
        penal: 惩罚参数
        rmin: 最小过滤半径
    """
    # 材料属性
    E0 = 1.0
    Emin = 1e-9
    nu = 0.3
    DH = E0 / (1 - nu**2) * np.array([[1, nu, 0], 
                                      [nu, 1, 0], 
                                      [0, 0, (1 - nu) / 2]])
    
    # 创建NURBS几何模型
    NURBS = geom_mod(L, W, Order, Num, BoundCon)
    
    # IGA预处理
    CtrPts, Ele, GauPts = pre_iga(NURBS)
    Dim = len(Order)
    Dofs = {}
    Dofs['Num'] = Dim * CtrPts['Num']
    
    # 边界条件
    DBoudary, F = boun_cond(CtrPts, BoundCon, NURBS, Dofs['Num'])
    
    # 初始化控制设计变量
    X = {}
    X['CtrPts'] = np.ones(CtrPts['Num'])
    
    # 准备高斯点坐标
    # MATLAB: GauPts.Cor = [reshape(GauPts.CorU',1,GauPts.Num); reshape(GauPts.CorV',1,GauPts.Num)];
    # MATLAB的reshape是按列填充的，所以需要按列顺序（Fortran顺序）展开
    # GauPts.CorU的形状是(Ele.Num, Ele.GauPtsNum)，转置后是(Ele.GauPtsNum, Ele.Num)
    # reshape按列填充，相当于先按列读取，然后转置成行向量
    GauPts['Cor'] = np.vstack([
        GauPts['CorU'].T.flatten('F'),  # Fortran顺序（列优先）
        GauPts['CorV'].T.flatten('F')   # Fortran顺序（列优先）
    ])
    
    # 计算高斯点的物理坐标和权重
    GauPts['PCor'] = []
    GauPts['Pw'] = []
    for i in range(GauPts['Num']):
        u_val = GauPts['Cor'][0, i]
        v_val = GauPts['Cor'][1, i]
        pt = NURBS.evaluate_single((u_val, v_val))
        GauPts['PCor'].append([pt[0] / pt[3], pt[1] / pt[3], pt[2] / pt[3]])
        GauPts['Pw'].append(pt[3])
    
    GauPts['PCor'] = np.array(GauPts['PCor']).T
    
    # 计算基函数矩阵R
    N_list = []
    id_list = []
    for i in range(GauPts['Num']):
        uv = np.array([[GauPts['Cor'][0, i]], [GauPts['Cor'][1, i]]])
        N, id_array = nrbbasisfun(uv, NURBS)
        N_list.append(N.flatten())
        id_list.append(id_array.flatten())
    
    # 构建稀疏矩阵R
    R_rows = []
    R_cols = []
    R_data = []
    for i, (N, ids) in enumerate(zip(N_list, id_list)):
        for j, ctrlpt_idx in enumerate(ids):
            R_rows.append(i)
            R_cols.append(int(ctrlpt_idx) - 1)  # 转换为0-based索引
            R_data.append(N[j])
    
    R = csr_matrix((R_data, (R_rows, R_cols)), 
                   shape=(GauPts['Num'], CtrPts['Num']))
    
    # 计算基函数导数
    dRu, dRv, dRu_id = nrbbasisfunder(GauPts['Cor'], NURBS)
    
    # 调试信息：打印nrbbasisfunder的输出信息
    print('=== nrbbasisfunder Debug Info ===')
    print(f'dRu shape: {dRu.shape}, dRv shape: {dRv.shape}, dRu_id shape: {dRu_id.shape}')
    print(f'GauPts.Num = {GauPts["Num"]}')
    if dRu.shape[0] > 0 and dRu.shape[1] > 0:
        print(f'dRu[0, 0:5] = {dRu[0, :min(5, dRu.shape[1])]}')
        print(f'dRv[0, 0:5] = {dRv[0, :min(5, dRv.shape[1])]}')
        print(f'dRu_id[0, 0:5] = {dRu_id[0, :min(5, dRu_id.shape[1])]}')
    print('=================================\n')
    
    # 初始化高斯点密度
    X['GauPts'] = (R @ X['CtrPts']).flatten()
    
    # 平滑机制
    Sh, Hs = shep_fun(CtrPts, rmin)
    
    # 开始优化循环
    change = 1.0
    nloop = 150
    Data = np.zeros((nloop, 2))
    Iter_Ch = np.zeros(nloop)
    
    # 准备绘图数据
    DenFied, Pos = plot_data(Num, NURBS)
    
    for loop in range(nloop):
        # IGA评估位移响应
        KE, dKE, dv_dg = stiff_ele2d(X, penal, Emin, DH, CtrPts, Ele, GauPts, dRu, dRv, dRu_id, NURBS)
        K = stiff_ass2d(KE, CtrPts, Ele, Dim, Dofs['Num'])
        U = solving(CtrPts, DBoudary, Dofs, K, F, BoundCon)
        
        # 目标函数和灵敏度分析
        J = 0.0
        dJ_dg = np.zeros(GauPts['Num'])
        
        for ide in range(Ele['Num']):
            Ele_NoCtPt = Ele['CtrPtsCon'][ide, :] - 1  # 转换为0-based索引
            edof = np.concatenate([Ele_NoCtPt, Ele_NoCtPt + CtrPts['Num']])
            Ue = U[edof.astype(int)]
            
            J += Ue.T @ KE[ide] @ Ue
            
            for i in range(Ele['GauPtsNum']):
                GptOrder = int(GauPts['Seque'][ide, i]) - 1  # 转换为0-based索引
                dJ_dg[GptOrder] = -Ue.T @ dKE[ide][i] @ Ue
        
        Data[loop, 0] = J
        Data[loop, 1] = np.mean(X['GauPts'])
        
        # 灵敏度过滤
        dJ_dp = R.T @ dJ_dg
        dJ_dp = Sh @ (dJ_dp / Hs)
        dv_dp = R.T @ dv_dg
        dv_dp = Sh @ (dv_dp / Hs)
        
        # 打印结果（在更新前）
        max_U = np.max(np.abs(U))
        max_F = np.max(np.abs(F))
        print(f' It.:{loop+1:5d} Obj.:{J:11.4f} Vol.:{np.mean(X["GauPts"]):7.3f} ch.:{change:7.3f} max|U|:{max_U:.2e} max|F|:{max_F:.2e}')

        # 优化准则更新设计变量（先更新再绘图，确保可视化的是最新设计）
        X = oc(X, R, Vmax, Sh, Hs, dJ_dp, dv_dp)
        change = np.max(np.abs(X['CtrPts_new'] - X['CtrPts']))
        Iter_Ch[loop] = change
        X['CtrPts'] = X['CtrPts_new']
        # 保障一致：基于最新控制点重算高斯点密度
        X['GauPts'] = (R @ X['CtrPts']).flatten()

        # 绘图（使用最新设计）
        X = plot_topy(X, GauPts, CtrPts, DenFied, Pos)

        if change < 0.01:
            break
    
    print(f'\n优化完成！最终迭代次数: {loop+1}')
    print(f'最终目标函数值: {J:.4f}')
    print(f'最终体积分数: {np.mean(X["GauPts"]):.4f}')

