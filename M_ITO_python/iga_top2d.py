"""
IGA 2D 拓扑优化主函数
从 MATLAB 代码转换
"""

import numpy as np
from scipy.sparse import csr_matrix
from nurbs import nrbeval, nrbbasisfun, nrbbasisfunder
from geom_mod import geom_mod
from pre_iga import pre_iga
from boun_cond import boun_cond
from shep_fun import shep_fun
from stiff_ele2d import stiff_ele2d
from stiff_ass2d import stiff_ass2d
from solving import solving
from oc import oc
from plot_data import plot_data
from plot_topy import plot_topy


def iga_top2d(L, W, Order, Num, BoundCon, Vmax, penal, rmin, case_number=None):
    """
    基于等几何分析的 2D 拓扑优化主函数
    
    参数:
        L: 长度
        W: 宽度
        Order: NURBS 次数提升 [Order_U, Order_V]
        Num: 控制点数量 [Num_U, Num_V]
        BoundCon: 边界条件类型 (1-5)
        Vmax: 最大体积分数
        penal: SIMP 惩罚因子
        rmin: 最小半径 (平滑)
    
    改编自 MATLAB IgaTop2D 代码
    """
    print('\n========== IgaTop2D START ==========')
    print(f'[INPUT] L={L:.2f}, W={W:.2f}, Order=[{Order[0]},{Order[1]}], Num=[{Num[0]},{Num[1]}]')
    print(f'[INPUT] BoundCon={BoundCon}, Vmax={Vmax:.3f}, penal={penal:.1f}, rmin={rmin:.1f}')
    
    # ========== 材料属性 ==========
    E0 = 1.0
    Emin = 1e-9
    nu = 0.3
    DH = E0 / (1 - nu**2) * np.array([[1, nu, 0], 
                                       [nu, 1, 0], 
                                       [0, 0, (1-nu)/2]])
    
    # ========== 生成几何模型 ==========
    NURBS = geom_mod(L, W, Order, Num, BoundCon)
    print(f'[NURBS] number=[{NURBS["number"][0]},{NURBS["number"][1]}], order=[{NURBS["order"][0]},{NURBS["order"][1]}]')
    print(f'[NURBS] knots[0] length={len(NURBS["knots"][0])}, range=[{NURBS["knots"][0][0]:.6f}, {NURBS["knots"][0][-1]:.6f}]')
    print(f'[NURBS] knots[1] length={len(NURBS["knots"][1])}, range=[{NURBS["knots"][1][0]:.6f}, {NURBS["knots"][1][-1]:.6f}]')
    
    # ========== IGA 准备 ==========
    CtrPts, Ele, GauPts = pre_iga(NURBS)
    Dim = len(NURBS['order'])
    Dofs = {'Num': Dim * CtrPts['Num']}
    print(f'[IGA] Dim={Dim}, Dofs.Num={Dofs["Num"]}')
    
    DBoudary, F = boun_cond(CtrPts, BoundCon, NURBS, Dofs['Num'])
    print(f'[LOAD] F nonzero={np.count_nonzero(F)}, sum(abs(F))={np.sum(np.abs(F)):.6e}, max(abs(F))={np.max(np.abs(F)):.6e}')
    
    # ========== 初始化设计变量 ==========
    print('\n初始化设计变量...')
    X = {}
    X['CtrPts'] = np.ones((CtrPts['Num'], 1))
    print(f'X.CtrPts 初始化: shape={X["CtrPts"].shape}, 全为1')
    
    # 准备高斯点坐标（与 MATLAB 的 reshape(A',1,...) 等价：在 NumPy 中为按 C 顺序展平原矩阵）
    # 与 MATLAB: [reshape(CorU',1,Num); reshape(CorV',1,Num)] 完全等价
    GauPts['Cor'] = np.vstack([
        GauPts['CorU'].T.flatten(order='F'),
        GauPts['CorV'].T.flatten(order='F')
    ])
    # 简要检查参数范围
    print(f"[Debug] Param U in GauPts.Cor: [{np.min(GauPts['Cor'][0,:]):.2f}, {np.max(GauPts['Cor'][0,:]):.2f}]")
    print(f"[Debug] Param V in GauPts.Cor: [{np.min(GauPts['Cor'][1,:]):.2f}, {np.max(GauPts['Cor'][1,:]):.2f}]")
    
    # 在高斯点求值 NURBS（使用理性基函数映射 R，避免散点求值路径差异）
    # 先按参数点计算理性基函数 N 与非零基函数索引 id_vals
    GauPts['PCor'] = None  # 占位，稍后用 R 与控制点直接计算物理坐标
    
    # 计算基函数
    print('\n计算基函数...')
    N, id_vals = nrbbasisfun(GauPts['Cor'], NURBS)
    GauPts['id_vals'] = id_vals  # 保存每个高斯点对应的非零基函数编号（1-based）
    
    print(f'N shape: {N.shape}, id_vals shape: {id_vals.shape}')
    print(f'N(1,:) = {N[0, :]}')
    print(f'id(1,:) = {id_vals[0, :]}')
    
    # 构建稀疏映射矩阵 R
    R = np.zeros((GauPts['Num'], CtrPts['Num']))
    for i in range(GauPts['Num']):
        R[i, id_vals[i, :] - 1] = N[i, :]  # 转为 0-based 索引
    R = csr_matrix(R)
    print(f'R 稀疏矩阵: shape={R.shape}, nnz={R.nnz}')

    # 用 R 与控制点坐标求高斯点物理坐标（与 MATLAB 一致，避免散点 nrbeval 差异）
    GauPts['PCor'] = np.vstack([
        (R @ CtrPts['Cordis'][0, :].reshape(-1, 1)).flatten(),
        (R @ CtrPts['Cordis'][1, :].reshape(-1, 1)).flatten(),
        np.zeros(GauPts['Num'])
    ])
    
    # 计算基函数导数
    dRu, dRv, id_dR = nrbbasisfunder(GauPts['Cor'], NURBS)
    GauPts['id_dR'] = id_dR
    print(f'[BASIS] dRu size=[{dRu.shape[0]},{dRu.shape[1]}], dRv size=[{dRv.shape[0]},{dRv.shape[1]}]')
    
    print(f'dRu shape: {dRu.shape}, dRv shape: {dRv.shape}')
    print(f'dRu(1,:) = {dRu[0, :]}')
    print(f'dRv(1,:) = {dRv[0, :]}')
    
    # 调试信息（已禁用）
    if False:
        print('=== nrbbasisfunder Debug Info ===')
        print(f'dRu size: [{dRu.shape[0]}, {dRu.shape[1]}]')
        print(f'dRv size: [{dRv.shape[0]}, {dRv.shape[1]}]')
        print(f'GauPts.Num = {GauPts["Num"]}')
        if dRu.shape[0] > 0 and dRu.shape[1] > 0:
            print(f'dRu[0, :5] = {dRu[0, :min(5, dRu.shape[1])]}')
            print(f'dRv[0, :5] = {dRv[0, :min(5, dRv.shape[1])]}')
        print('=================================\n')
    
    # 映射到高斯点
    X['GauPts'] = R @ X['CtrPts']
    
    # ========== 平滑机制 ==========
    print('构建平滑机制...')
    Sh, Hs = shep_fun(CtrPts, rmin)
    
    # ========== 准备绘图 ==========
    print('准备绘图数据...')
    DenFied, Pos = plot_data(Num, NURBS)
    
    # ========== 优化循环 ==========
    print('\n开始优化迭代...')
    print('='*60)
    change = 1.0
    nloop = 150
    Data = np.zeros((nloop, 2))
    Iter_Ch = np.zeros(nloop)
    
    for loop in range(nloop):
        # IGA 求解位移响应
        KE, dKE, dv_dg = stiff_ele2d(X, penal, Emin, DH, CtrPts, Ele, GauPts, dRu, dRv)
        
        if loop == 0:
            print(f'\n=== 第1次迭代详细信息 ===')
            print(f'KE{{1}} shape: {KE[0].shape}')
            print(f'KE{{1}}(1:3,1:3):')
            for i in range(3):
                print(f'  {KE[0][i, :3]}')
            print(f'dv_dg(1:5) = {dv_dg[:5, 0]}')
        
        K = stiff_ass2d(KE, CtrPts, Ele, Dim, Dofs['Num'])
        
        if loop == 0:
            print(f'K: shape={K.shape}, nnz={K.nnz}')
        
        U = solving(CtrPts, DBoudary, Dofs, K, F, BoundCon)
        
        if loop == 0:
            print(f'U: shape={U.shape}')
            print(f'U(1:10) = {U[:10, 0]}')
            print(f'max(abs(U)) = {np.max(np.abs(U)):.10e}')
        
        # 目标函数和灵敏度分析
        J = 0.0
        dJ_dg = np.zeros((GauPts['Num'], 1))
        
        for ide in range(Ele['Num']):
            Ele_NoCtPt = Ele['CtrPtsCon'][ide, :]  # 1-based
            edof = np.concatenate([Ele_NoCtPt - 1, Ele_NoCtPt - 1 + CtrPts['Num']])  # 转为 0-based
            Ue = U[edof, 0]
            J += Ue.T @ KE[ide] @ Ue
            
            for i in range(Ele['GauPtsNum']):
                GptOrder = GauPts['Seque'][ide, i] - 1  # 转为 0-based
                dJ_dg[GptOrder] = -Ue.T @ dKE[ide][i] @ Ue
        
        Data[loop, 0] = J
        Data[loop, 1] = np.mean(X['GauPts'])
        print(f'[OBJ] J={J:.6e}, mean(X.GauPts)={np.mean(X["GauPts"]):.6f}')
        print(f'[SENS] dJ_dg: min={np.min(dJ_dg):.6e}, max={np.max(dJ_dg):.6e}, sum={np.sum(dJ_dg):.6e}')
        
        # 链式法则计算灵敏度
        dJ_dp = R.T @ dJ_dg
        dJ_dp = (Sh @ dJ_dp) / Hs.reshape(-1, 1)
        
        dv_dp = R.T @ dv_dg
        dv_dp = (Sh @ dv_dp) / Hs.reshape(-1, 1)
        print(f'[SENS] dJ_dp: min={np.min(dJ_dp):.6e}, max={np.max(dJ_dp):.6e}, sum={np.sum(dJ_dp):.6e}')
        print(f'[SENS] dv_dp: min={np.min(dv_dp):.6e}, max={np.max(dv_dp):.6e}, sum={np.sum(dv_dp):.6e}')
        
        # 打印和绘图
        print(f' It.:{loop+1:5d} Obj.:{J:11.4f} Vol.:{np.mean(X["GauPts"]):7.3f} ch.:{change:7.3f}')
        X = plot_topy(X, GauPts, CtrPts, DenFied, Pos, case_number=case_number)
        
        if change < 0.01:
            X = plot_topy(X, GauPts, CtrPts, DenFied, Pos, case_number=case_number)
            break
        
        # 优化准则更新设计变量
        X = oc(X, R, Vmax, Sh, Hs, dJ_dp, dv_dp)
        print(f'[OC] X.CtrPts_new: min={np.min(X["CtrPts_new"]):.6f}, max={np.max(X["CtrPts_new"]):.6f}, mean={np.mean(X["CtrPts_new"]):.6f}')
        change = np.max(np.abs(X['CtrPts_new'] - X['CtrPts']))
        print(f'[CHANGE] change={change:.6e}')
        Iter_Ch[loop] = change
        X['CtrPts'] = X['CtrPts_new']
    
    print('\n========== IgaTop2D END ==========')
    
    return X, Data, Iter_Ch


# ======================================================================================================================
# 函数: iga_top2d
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
