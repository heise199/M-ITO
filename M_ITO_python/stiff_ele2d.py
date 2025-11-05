"""
2D 单元刚度矩阵 (2D Element Stiffness Matrix)
从 MATLAB 代码转换
"""

import numpy as np


def stiff_ele2d(X, penal, Emin, DH, CtrPts, Ele, GauPts, dRu, dRv):
    """
    计算单元刚度矩阵及其导数
    
    参数:
        X: 设计变量字典
           - GauPts: 高斯点处的设计变量
        penal: SIMP 惩罚因子
        Emin: 最小杨氏模量
        DH: 弹性矩阵
        CtrPts: 控制点信息
        Ele: 单元信息
        GauPts: 高斯积分点信息
        dRu: U 方向的基函数导数
        dRv: V 方向的基函数导数
    
    返回:
        KE: 单元刚度矩阵列表
        dKE: 单元刚度矩阵导数列表
        dv_dg: 体积导数
    
    改编自 MATLAB IgaTop2D 代码
    """
    KE = []
    dKE = []
    dv_dg = np.zeros((GauPts['Num'], 1))
    Nen = Ele['CtrPtsNum']
    
    # 调试信息（仅第一次）
    if False:  # 设为True可启用调试
        print('=== Stiff_Ele2D Debug Info ===')
        print(f'Ele.Num = {Ele["Num"]}, Ele.CtrPtsNum = {Ele["CtrPtsNum"]}, GauPts.Num = {GauPts["Num"]}')
        print(f'dRu size: [{dRu.shape[0]}, {dRu.shape[1]}], dRv size: [{dRv.shape[0]}, {dRv.shape[1]}]')
        print(f'CtrPts.Cordis size: [{CtrPts["Cordis"].shape[0]}, {CtrPts["Cordis"].shape[1]}]')
    
    neg_det_count_J1 = 0
    neg_det_count_J = 0
    
    for ide in range(Ele['Num']):
        # 直接根据线性编号计算 (idu, idv) 保持一致顺序：idu 变化最快
        idu = ide % Ele['NumU']
        idv = ide // Ele['NumU']
        Ele_Knot_U = Ele['KnotsU'][idu, :]
        Ele_Knot_V = Ele['KnotsV'][idv, :]
        Ele_NoCtPt = Ele['CtrPtsCon'][ide, :]  # 1-based 索引
        
        # 调试：检查维度（已禁用）
        if False and ide == 0:
            print(f'  [Debug] CtrPts["Cordis"].shape = {CtrPts["Cordis"].shape}')
            print(f'  [Debug] Ele_NoCtPt = {Ele_NoCtPt}')
            print(f'  [Debug] Ele_NoCtPt.shape = {Ele_NoCtPt.shape}')
            print(f'  [Debug] Ele_NoCtPt - 1 = {Ele_NoCtPt - 1}')
        
        # 正确的索引方式：Ele_NoCtPt - 1 应该作为整数数组
        Ele_CoCtPt = CtrPts['Cordis'][0:2, :][:, Ele_NoCtPt - 1]  # 转为 0-based，提取 XY 坐标
        
        # 调试信息 (前几个单元或每 500 个单元)（已禁用）
        if False and (ide < 3 or (ide + 1) % 500 == 0):
            print(f'\n--- Element {ide + 1} ---')
            print(f'  idu={idu}, idv={idv}')
            print(f'  Ele_Knot_U = [{Ele_Knot_U[0]:.6f}, {Ele_Knot_U[1]:.6f}]')
            print(f'  Ele_Knot_V = [{Ele_Knot_V[0]:.6f}, {Ele_Knot_V[1]:.6f}]')
            print(f'  Ele_NoCtPt (first 5): {Ele_NoCtPt[:min(5, len(Ele_NoCtPt))]}')
            print(f'  Ele_CoCtPt size: [{Ele_CoCtPt.shape[0]}, {Ele_CoCtPt.shape[1]}]')
        
        Ke = np.zeros((2*Nen, 2*Nen))
        dKe = []
        
        for i in range(Ele['GauPtsNum']):
            GptOrder = GauPts['Seque'][ide, i] - 1  # 转为 0-based 索引
            
            # 调试信息已完全禁用
            
            # 提取并重排基函数导数 (相对于参数坐标)
            # 关键一致性：dRu/dRv 的列顺序必须与 Ele_NoCtPt 的非零基函数顺序一致
            # 我们用 GauPts['id_vals'] 映射（1-based），构建从基函数编号到列索引的查找表
            # 优先使用与 dR 同源的列顺序（来源于 nrbbasisfunder）
            ids_at_gp = GauPts.get('id_dR', GauPts.get('id_vals', None))
            if ids_at_gp is not None:
                ids_row = ids_at_gp[GptOrder, :].astype(int)  # 1-based
                # 建立编号到列下标的映射
                pos_map = {int(bid): idx for idx, bid in enumerate(ids_row, start=0)}
                # 如果某个 Ele_NoCtPt 不在该高斯点的支撑中，说明顺序映射不一致；
                # 采取保守回退：使用 ids_row 自身顺序（与 dR 列一致），并在最后一次性重排到 Ele_NoCtPt 顺序。
                if any(int(bid) not in pos_map for bid in Ele_NoCtPt):
                    col_idx = list(range(len(ids_row)))
                    # 同时覆盖 Ele_CoCtPt 以匹配该顺序
                    Ele_CoCtPt = CtrPts['Cordis'][0:2, :][:, ids_row - 1]
                else:
                    col_idx = [pos_map[int(bid)] for bid in Ele_NoCtPt]
                dR_dPara = np.vstack([
                    dRu[GptOrder, col_idx],
                    dRv[GptOrder, col_idx]
                ])
            else:
                # 回退：假设顺序已一致
                dR_dPara = np.vstack([dRu[GptOrder, :], dRv[GptOrder, :]])
            
            # 计算雅可比矩阵 J1: 从参数空间到物理空间
            dPhy_dPara = dR_dPara @ Ele_CoCtPt.T
            J1 = dPhy_dPara
            
            # 检查行列式
            det_J1 = np.linalg.det(J1)
            # 确保 det_J1 是标量
            if np.ndim(det_J1) > 0:
                if det_J1.size == 1:
                    det_J1 = det_J1.item()
                else:
                    # 如果是数组，取第一个元素或平均值
                    det_J1 = np.mean(det_J1) if det_J1.size > 0 else 1.0
            if float(det_J1) < 0:
                neg_det_count_J1 += 1
                if neg_det_count_J1 <= 5 or (ide < 3 and i < 2):
                    print(f'  [Element {ide + 1}, Gauss point {i + 1}] J1 determinant is negative: det={det_J1:.6e}')
                    print(f'    J1 = [{J1[0,0]:.6e}, {J1[0,1]:.6e}; {J1[1,0]:.6e}, {J1[1,1]:.6e}]')
            
            # 计算基函数导数 (相对于物理坐标)
            try:
                # 尝试使用标准求逆
                if abs(det_J1) > 1e-10:
                    dR_dPhy = np.linalg.inv(J1) @ dR_dPara
                else:
                    # 使用伪逆处理奇异或接近奇异的情况
                    dR_dPhy = np.linalg.pinv(J1) @ dR_dPara
            except np.linalg.LinAlgError:
                # 如果求逆失败，使用伪逆
                dR_dPhy = np.linalg.pinv(J1) @ dR_dPara
            
            
            # 构建应变-位移矩阵 B
            Be = np.zeros((3, 2*Nen))
            Be[0, 0:Nen] = dR_dPhy[0, :]
            Be[1, Nen:2*Nen] = dR_dPhy[1, :]
            Be[2, 0:Nen] = dR_dPhy[1, :]
            Be[2, Nen:2*Nen] = dR_dPhy[0, :]
            
            # 从父空间到参数空间的雅可比
            dPara_dPare = np.zeros((2, 2))
            dPara_dPare[0, 0] = (Ele_Knot_U[1] - Ele_Knot_U[0]) / 2
            dPara_dPare[1, 1] = (Ele_Knot_V[1] - Ele_Knot_V[0]) / 2
            J2 = dPara_dPare
            
            # 从物理空间到父空间的雅可比
            J = J1 @ J2
            
            # 检查行列式
            det_J = np.linalg.det(J)
            # 确保 det_J 是标量
            if np.ndim(det_J) > 0:
                if det_J.size == 1:
                    det_J = det_J.item()
                else:
                    # 如果是数组，取第一个元素或平均值
                    det_J = np.mean(det_J) if det_J.size > 0 else 1.0
            if float(det_J) < 0:
                neg_det_count_J += 1
                if neg_det_count_J <= 5 or (ide < 3 and i < 2):
                    print(f'  [Element {ide + 1}, Gauss point {i + 1}] J determinant is negative: det={det_J:.6e}')
                    print(f'    J2 = [{J2[0,0]:.6e}, 0; 0, {J2[1,1]:.6e}]')
            
            # 权重因子
            weight = GauPts['Weigh'][i] * det_J
            
            # 材料插值 (SIMP)
            E = Emin + X['GauPts'][GptOrder, 0]**penal * (1 - Emin)
            
            # 单元刚度矩阵
            Ke += E * weight * (Be.T @ DH @ Be)
            
            # 单元刚度矩阵导数
            dKe_i = penal * X['GauPts'][GptOrder, 0]**(penal-1) * (1 - Emin) * weight * (Be.T @ DH @ Be)
            dKe.append(dKe_i)
            
            # 体积导数
            dv_dg[GptOrder, 0] = weight
        
        KE.append(Ke)
        dKE.append(dKe)
    
    # 打印统计信息
    # 统计负行列式数量（仅在有问题时显示）
    if neg_det_count_J1 > 0 or neg_det_count_J > 0:
        print(f'\n[Warning] Negative determinants found!')
        print(f'  J1: {neg_det_count_J1} / {Ele["Num"]*Ele["GauPtsNum"]}')
        print(f'  J: {neg_det_count_J} / {Ele["Num"]*Ele["GauPtsNum"]}\n')
    
    return KE, dKE, dv_dg


# ======================================================================================================================
# 函数: stiff_ele2d
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
