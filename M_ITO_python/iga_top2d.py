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
    
    # ========== Material properties ==========
    E0 = 1.0
    Emin = 1e-9
    nu = 0.3
    DH = E0 / (1 - nu**2) * np.array([[1, nu, 0], 
                                       [nu, 1, 0], 
                                       [0, 0, (1-nu)/2]])
    
    # ========== Generate geometry model ==========
    NURBS = geom_mod(L, W, Order, Num, BoundCon)
    print(f'[NURBS] number=[{NURBS["number"][0]},{NURBS["number"][1]}], order=[{NURBS["order"][0]},{NURBS["order"][1]}]')
    print(f'[NURBS] knots{{1}} length={len(NURBS["knots"][0])}, range=[{NURBS["knots"][0][0]:.6f}, {NURBS["knots"][0][-1]:.6f}]')
    print(f'[NURBS] knots{{2}} length={len(NURBS["knots"][1])}, range=[{NURBS["knots"][1][0]:.6f}, {NURBS["knots"][1][-1]:.6f}]')
    
    # ========== Preparation for IGA ==========
    CtrPts, Ele, GauPts = pre_iga(NURBS)
    Dim = len(NURBS['order'])
    Dofs = {'Num': Dim * CtrPts['Num']}
    print(f'[IGA] Dim={Dim}, Dofs.Num={Dofs["Num"]}')
    
    DBoudary, F = boun_cond(CtrPts, BoundCon, NURBS, Dofs['Num'])
    print(f'[LOAD] F nonzero={np.count_nonzero(F)}, sum(abs(F))={np.sum(np.abs(F)):.6e}, max(abs(F))={np.max(np.abs(F)):.6e}')
    
    # ========== Initialization of control design variables ==========
    X = {}
    X['CtrPts'] = np.ones((CtrPts['Num'], 1))
    
    # Prepare Gauss point coordinates
    # MATLAB: GauPts.Cor = [reshape(GauPts.CorU',1,GauPts.Num); reshape(GauPts.CorV',1,GauPts.Num)];
    GauPts['Cor'] = np.vstack([
        GauPts['CorU'].flatten(order='C'),  # Flatten row-wise to match MATLAB reshape after transpose
        GauPts['CorV'].flatten(order='C')
    ])
    
    # Evaluate NURBS at Gauss points
    GauPts['PCor'], GauPts['Pw'] = nrbeval(NURBS, GauPts['Cor'])
    GauPts['PCor'] = GauPts['PCor'] / GauPts['Pw']
    
    # Compute basis functions
    N, id_vals = nrbbasisfun(GauPts['Cor'], NURBS)
    GauPts['id_vals'] = id_vals
    
    # Build sparse mapping matrix R
    R = np.zeros((GauPts['Num'], CtrPts['Num']))
    for i in range(GauPts['Num']):
        R[i, id_vals[i, :] - 1] = N[i, :]  # Convert to 0-based index
    R = csr_matrix(R)
    print(f'[BASIS] N size=[{N.shape[0]},{N.shape[1]}], id size=[{id_vals.shape[0]},{id_vals.shape[1]}]')
    print(f'[BASIS] R size=[{R.shape[0]},{R.shape[1]}], nnz(R)={R.nnz}')
    
    # Compute basis function derivatives
    dRu, dRv, id_dR = nrbbasisfunder(GauPts['Cor'], NURBS)
    GauPts['id_dR'] = id_dR
    print(f'[BASIS] dRu size=[{dRu.shape[0]},{dRu.shape[1]}], dRv size=[{dRv.shape[0]},{dRv.shape[1]}]')
    
    # Debug info: print nrbbasisfunder output information
    print('=== nrbbasisfunder Debug Info ===')
    print(f'dRu size: [{dRu.shape[0]}, {dRu.shape[1]}]')
    print(f'dRv size: [{dRv.shape[0]}, {dRv.shape[1]}]')
    print(f'GauPts.Num = {GauPts["Num"]}')
    if dRu.shape[0] > 0 and dRu.shape[1] > 0:
        # Format array output to match MATLAB's mat2str format (space-separated, brackets)
        dRu_vals = dRu[0, :min(5, dRu.shape[1])]
        dRv_vals = dRv[0, :min(5, dRv.shape[1])]
        # MATLAB mat2str format: [val1 val2 val3 ...] (space-separated)
        dRu_str = '[' + ' '.join([f'{x:.6f}' for x in dRu_vals]) + ']'
        dRv_str = '[' + ' '.join([f'{x:.6f}' for x in dRv_vals]) + ']'
        print(f'dRu(1, 1:5) = {dRu_str}')
        print(f'dRv(1, 1:5) = {dRv_str}')
    print('=================================')
    print()
    
    # Map to Gauss points
    X['GauPts'] = R @ X['CtrPts']
    
    # ========== Smoothing mechanism ==========
    Sh, Hs = shep_fun(CtrPts, rmin)
    
    # ========== Prepare plotting ==========
    DenFied, Pos = plot_data(Num, NURBS)
    
    # ========== Start optimization loop ==========
    change = 1.0
    nloop = 150
    Data = np.zeros((nloop, 2))
    Iter_Ch = np.zeros(nloop)
    
    for loop in range(nloop):
        print(f'\n========== ITERATION {loop+1} ==========')
        # IGA to evaluate the displacement responses
        KE, dKE, dv_dg = stiff_ele2d(X, penal, Emin, DH, CtrPts, Ele, GauPts, dRu, dRv)
        print(f'[STIFF] KE{{1}} size=[{KE[0].shape[0]},{KE[0].shape[1]}], norm(KE{{1}})={np.linalg.norm(KE[0], "fro"):.6e}')
        print(f'[STIFF] dv_dg: min={np.min(dv_dg):.6e}, max={np.max(dv_dg):.6e}, sum={np.sum(dv_dg):.6e}')
        
        K = stiff_ass2d(KE, CtrPts, Ele, Dim, Dofs['Num'])
        print(f'[ASS] K size=[{K.shape[0]},{K.shape[1]}], nnz(K)={K.nnz}, norm(K)={np.linalg.norm(K.toarray(), "fro"):.6e}')
        
        U = solving(CtrPts, DBoudary, Dofs, K, F, BoundCon)
        print(f'[SOLVE] U: min={np.min(U):.6e}, max={np.max(U):.6e}, norm={np.linalg.norm(U):.6e}')
        
        # Objective function and sensitivity analysis
        J = 0.0
        dJ_dg = np.zeros((GauPts['Num'], 1))
        
        for ide in range(Ele['Num']):
            Ele_NoCtPt = Ele['CtrPtsCon'][ide, :]  # 1-based
            edof = np.concatenate([Ele_NoCtPt - 1, Ele_NoCtPt - 1 + CtrPts['Num']])  # Convert to 0-based
            Ue = U[edof, 0]
            J += Ue.T @ KE[ide] @ Ue
            
            for i in range(Ele['GauPtsNum']):
                GptOrder = GauPts['Seque'][ide, i] - 1  # Convert to 0-based
                dJ_dg[GptOrder] = -Ue.T @ dKE[ide][i] @ Ue
        
        Data[loop, 0] = J
        Data[loop, 1] = np.mean(X['GauPts'])
        print(f'[OBJ] J={J:.6e}, mean(X.GauPts)={np.mean(X["GauPts"]):.6f}')
        print(f'[SENS] dJ_dg: min={np.min(dJ_dg):.6e}, max={np.max(dJ_dg):.6e}, sum={np.sum(dJ_dg):.6e}')
        
        # Chain rule for sensitivity
        dJ_dp = R.T @ dJ_dg
        dJ_dp = (Sh @ dJ_dp) / Hs.reshape(-1, 1)
        
        dv_dp = R.T @ dv_dg
        dv_dp = (Sh @ dv_dp) / Hs.reshape(-1, 1)
        print(f'[SENS] dJ_dp: min={np.min(dJ_dp):.6e}, max={np.max(dJ_dp):.6e}, sum={np.sum(dJ_dp):.6e}')
        print(f'[SENS] dv_dp: min={np.min(dv_dp):.6e}, max={np.max(dv_dp):.6e}, sum={np.sum(dv_dp):.6e}')
        
        # Print and plot results
        # Match MATLAB format: fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n',loop,J,mean(X.GauPts(:)),change);
        print(f' It.:{loop+1:5d} Obj.:{J:11.4f} Vol.:{np.mean(X["GauPts"]):7.3f} ch.:{change:7.3f}')
        X = plot_topy(X, GauPts, CtrPts, DenFied, Pos, case_number=case_number)
        
        if change < 0.01:
            X = plot_topy(X, GauPts, CtrPts, DenFied, Pos, case_number=case_number)
            break
        
        # Optimality criteria to update design variables
        X = oc(X, R, Vmax, Sh, Hs, dJ_dp, dv_dp)
        print(f'[OC] X.CtrPts_new: min={np.min(X["CtrPts_new"]):.6f}, max={np.max(X["CtrPts_new"]):.6f}, mean={np.mean(X["CtrPts_new"]):.6f}')
        change = np.max(np.abs(X['CtrPts_new'] - X['CtrPts']))
        Iter_Ch[loop] = change
        print(f'[CHANGE] change={change:.6e}')
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
