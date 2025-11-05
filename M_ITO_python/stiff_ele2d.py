"""
单元刚度矩阵模块 - 计算单元刚度矩阵和导数
"""
import numpy as np
from nrbbasisfunder import nrbbasisfunder


def stiff_ele2d(X, penal, Emin, DH, CtrPts, Ele, GauPts, dRu, dRv, dRu_id, NURBS):
    """
    计算单元刚度矩阵和导数
    
    参数:
        X: 设计变量
        penal: 惩罚参数
        Emin: 最小弹性模量
        DH: 材料矩阵
        CtrPts: 控制点信息
        Ele: 单元信息
        GauPts: 高斯点信息
        dRu: u方向的基函数导数
        dRv: v方向的基函数导数
    
    返回:
        KE: 单元刚度矩阵列表
        dKE: 单元刚度矩阵导数列表
        dv_dg: 体积对密度的导数
    """
    KE = []
    dKE = []
    dv_dg = np.zeros(GauPts['Num'])
    
    Nen = Ele['CtrPtsNum']
    
    # 调试信息：打印基本参数
    print('=== Stiff_Ele2D Debug Info ===')
    print(f'Ele.Num = {Ele["Num"]}, Ele.CtrPtsNum = {Ele["CtrPtsNum"]}, GauPts.Num = {GauPts["Num"]}')
    print(f'dRu shape: {dRu.shape}, dRv shape: {dRv.shape}')
    print(f'CtrPts.Cordis shape: {CtrPts["Cordis"].shape}')
    
    neg_det_count_J1 = 0
    neg_det_count_J = 0
    
    for ide in range(Ele['Num']):
        # 找到单元在两个参数方向上的索引
        idv, idu = np.where(Ele['Seque'] == ide + 1)
        idu = idu[0]
        idv = idv[0]
        
        Ele_Knot_U = Ele['KnotsU'][idu, :]
        Ele_Knot_V = Ele['KnotsV'][idv, :]
        Ele_NoCtPt = Ele['CtrPtsCon'][ide, :] - 1  # 转换为0-based索引
        Ele_NoCtPt = Ele_NoCtPt.astype(int)
        
        # MATLAB: Ele_NoCtPt = Ele.CtrPtsCon(ide,:);
        # Ele_NoCtPt是MATLAB索引（从1开始），包含该单元的控制点索引
        Ele_NoCtPt_matlab = Ele['CtrPtsCon'][ide, :]  # MATLAB索引，从1开始
        
        # MATLAB: Ele_CoCtPt = CtrPts.Cordis(1:2,Ele_NoCtPt);
        # 为彻底消除展平顺序可能的不一致，这里直接从 NURBS.coefs 依据列优先（MATLAB）索引恢复坐标
        # 将MATLAB索引 Ele_NoCtPt_matlab（1-based, 列优先）转换为 (u_idx, v_idx)，再用 coefs 取值
        num_u = NURBS.ctrlpts_size_u
        num_v = NURBS.ctrlpts_size_v
        Ele_CoCtPt = np.zeros((2, len(Ele_NoCtPt_matlab)))
        for j, ctrlpt_id in enumerate(Ele_NoCtPt_matlab.astype(int)):
            idx0 = int(ctrlpt_id) - 1  # 0-based
            u_idx = idx0 % num_u
            v_idx = idx0 // num_u
            w_pt = NURBS.coefs[3, u_idx, v_idx]
            if abs(w_pt) > 1e-15:
                Ele_CoCtPt[0, j] = NURBS.coefs[0, u_idx, v_idx] / w_pt
                Ele_CoCtPt[1, j] = NURBS.coefs[1, u_idx, v_idx] / w_pt
            else:
                Ele_CoCtPt[0, j] = NURBS.coefs[0, u_idx, v_idx]
                Ele_CoCtPt[1, j] = NURBS.coefs[1, u_idx, v_idx]
        
        # 实际的控制点数量
        num_ctrlpts = len(Ele_NoCtPt_matlab)
        
        Ke = np.zeros((2 * num_ctrlpts, 2 * num_ctrlpts))
        dKe = []
        
        for i in range(Ele['GauPtsNum']):
            GptOrder = int(GauPts['Seque'][ide, i]) - 1  # 转换为0-based索引
            
            # MATLAB: dR_dPara = [dRu(GptOrder,:); dRv(GptOrder,:)];
            # 关键修复：需要根据该高斯点的实际控制点索引顺序，重新排列dRu/dRv的列
            # 使得列顺序与Ele.CtrPtsCon(ide,:)的顺序一致
            
            # 获取该高斯点的参数坐标
            gpt_u = GauPts['Cor'][0, GptOrder]
            gpt_v = GauPts['Cor'][1, GptOrder]
            
            # 计算该高斯点的控制点索引（与nrbbasisfun中的计算方式一致）
            from utils import find_span
            knotvector_u = np.array(NURBS.knotvector_u)
            knotvector_v = np.array(NURBS.knotvector_v)
            
            span_u = find_span(NURBS.ctrlpts_size_u - 1, NURBS.degree_u, gpt_u, knotvector_u)
            span_v = find_span(NURBS.ctrlpts_size_v - 1, NURBS.degree_v, gpt_v, knotvector_v)
            
            # 计算该高斯点的控制点索引顺序（MATLAB索引，从1开始）
            # MATLAB的sub2ind([num_u, num_v], u, v) = (v-1)*num_u + u（列优先！）
            gpt_ctrlpt_ids = []
            for v_idx in range(span_v - NURBS.degree_v, span_v + 1):
                for u_idx in range(span_u - NURBS.degree_u, span_u + 1):
                    if v_idx >= 0 and v_idx < NURBS.ctrlpts_size_v and \
                       u_idx >= 0 and u_idx < NURBS.ctrlpts_size_u:
                        # MATLAB索引（从1开始）：(v_idx_matlab - 1) * num_u + u_idx_matlab
                        ctrlpt_id_matlab = v_idx * NURBS.ctrlpts_size_u + (u_idx + 1)
                        gpt_ctrlpt_ids.append(ctrlpt_id_matlab)
            
            # 根据Ele.CtrPtsCon(ide,:)的顺序，从dRu/dRv中提取对应的列
            # 关键：dRu(GptOrder,:)的列顺序对应dRu_id(GptOrder,:)的顺序
            # 我们必须根据Ele.CtrPtsCon中每个控制点在dRu_id(GptOrder,:)中的位置重新映射
            
            # 使用dRu_id直接获取该高斯点的控制点索引顺序（更准确）
            gpt_ctrlpt_ids_from_dRu = dRu_id[GptOrder, :].tolist()
            
            dRu_row = np.zeros(num_ctrlpts)
            dRv_row = np.zeros(num_ctrlpts)
            
            for j, ctrlpt_id_matlab in enumerate(Ele_NoCtPt_matlab):
                # 在dRu_id(GptOrder,:)中找到该控制点的位置
                if ctrlpt_id_matlab in gpt_ctrlpt_ids_from_dRu:
                    col_idx = gpt_ctrlpt_ids_from_dRu.index(ctrlpt_id_matlab)
                    if col_idx < dRu.shape[1]:
                        dRu_row[j] = dRu[GptOrder, col_idx]
                        dRv_row[j] = dRv[GptOrder, col_idx]
                    else:
                        # 列索引超出范围，设为0
                        dRu_row[j] = 0.0
                        dRv_row[j] = 0.0
                else:
                    # 如果找不到，说明该控制点不影响这个高斯点，设为0
                    dRu_row[j] = 0.0
                    dRv_row[j] = 0.0
            
            # 调试信息：打印前几个单元的信息（用于对比MATLAB）
            if (ide < 3 and i < 4) or (ide == 2 and i == 0):
                print(f'  [单元{ide}, 高斯点{i}] GptOrder={GptOrder}')
                print(f'    gpt_u={gpt_u:.6f}, gpt_v={gpt_v:.6f}')
                print(f'    span_u={span_u}, span_v={span_v}')
                print(f'    gpt_ctrlpt_ids (computed) = {gpt_ctrlpt_ids}')
                print(f'    gpt_ctrlpt_ids_from_dRu = {gpt_ctrlpt_ids_from_dRu}')
                print(f'    Ele_NoCtPt_matlab = {Ele_NoCtPt_matlab}')
                print(f'    dRu[GptOrder, :] = {dRu[GptOrder, :]}')
                print(f'    dRv[GptOrder, :] = {dRv[GptOrder, :]}')
                print(f'    dRu_row = {dRu_row}')
                print(f'    dRv_row = {dRv_row}')
                
                # 检查是否所有值都为0或大部分为0
                if np.allclose(dRu_row, 0) and np.allclose(dRv_row, 0):
                    print(f'    WARNING: All values are zero!')
                    print(f'    Checking if Ele_NoCtPt_matlab matches gpt_ctrlpt_ids_from_dRu...')
                    for j, ctrlpt_id in enumerate(Ele_NoCtPt_matlab):
                        in_list = ctrlpt_id in gpt_ctrlpt_ids_from_dRu
                        print(f'      Control point {ctrlpt_id}: in dRu_id = {in_list}')
                elif np.sum(np.abs(dRu_row)) < 1e-10 or np.sum(np.abs(dRv_row)) < 1e-10:
                    print(f'    WARNING: Most values are zero!')
                    print(f'    dRu_row sum(abs) = {np.sum(np.abs(dRu_row))}')
                    print(f'    dRv_row sum(abs) = {np.sum(np.abs(dRv_row))}')
                    print(f'    Checking mapping...')
                    for j, ctrlpt_id in enumerate(Ele_NoCtPt_matlab):
                        in_list = ctrlpt_id in gpt_ctrlpt_ids_from_dRu
                        if in_list:
                            col_idx = gpt_ctrlpt_ids_from_dRu.index(ctrlpt_id)
                            print(f'      Control point {ctrlpt_id}: in dRu_id = True, col_idx={col_idx}, dRu_val={dRu[GptOrder, col_idx]:.6e}, dRv_val={dRv[GptOrder, col_idx]:.6e}')
                        else:
                            print(f'      Control point {ctrlpt_id}: in dRu_id = False')
            
            dR_dPara = np.vstack([dRu_row, dRv_row])  # 2xNen
            
            # MATLAB: Ele_CoCtPt = CtrPts.Cordis(1:2,Ele_NoCtPt);
            # Ele_CoCtPt是2xNen的矩阵，第一行是x坐标，第二行是y坐标
            # 确保Ele_CoCtPt的形状正确
            if Ele_CoCtPt.shape[0] != 2:
                raise ValueError(f"Ele_CoCtPt的第一维应该是2，但得到{Ele_CoCtPt.shape[0]}")
            if Ele_CoCtPt.shape[1] != num_ctrlpts:
                raise ValueError(f"Ele_CoCtPt的第二维应该是{num_ctrlpts}，但得到{Ele_CoCtPt.shape[1]}")
            
            # MATLAB: dPhy_dPara = dR_dPara*Ele_CoCtPt';
            # dR_dPara是2xNen，Ele_CoCtPt'是Nenx2，结果是2x2
            # 这是雅可比矩阵：∂(x,y)/∂(u,v)
            dPhy_dPara = dR_dPara @ Ele_CoCtPt.T  # 2x2
            J1 = dPhy_dPara
            
            # 检查雅可比矩阵的行列式
            det_J1 = np.linalg.det(J1)
            
            # 调试信息：打印前几个单元的信息（用于对比MATLAB）
            if (ide < 2 and i == 0) or (ide == 0 and i < 2):
                print(f'  [单元{ide}, 高斯点{i}] J1行列式 = {det_J1:.6e}')
                print(f'    J1 = \n{J1}')
                print(f'    dR_dPara (前3列) = \n{dR_dPara[:, :min(3, dR_dPara.shape[1])]}')
                print(f'    Ele_CoCtPt (前3列) = \n{Ele_CoCtPt[:, :min(3, Ele_CoCtPt.shape[1])]}')
            
            if abs(det_J1) < 1e-10:
                # 如果行列式接近0，说明雅可比矩阵奇异
                print(f"错误: 单元{ide}高斯点{i}的雅可比矩阵J1奇异: det={det_J1}")
                print(f"  J1={J1}")
                print(f"  Ele_NoCtPt={Ele['CtrPtsCon'][ide, :]}")
                print(f"  dR_dPara=\n{dR_dPara}")
                print(f"  Ele_CoCtPt=\n{Ele_CoCtPt}")
                raise ValueError(f"雅可比矩阵奇异，行列式={det_J1}")
            
            # MATLAB代码中没有检查行列式符号，直接使用det(J)
            # 如果行列式为负，表示坐标方向反转，这在某些几何情况下是正常的
            # 但为了计算的连续性，我们使用绝对值（与MATLAB保持一致的行为）
            # 注意：MATLAB的det函数总是返回正值或负值，不会自动取绝对值
            # 但在这个应用中，行列式的符号不影响体积计算（因为用的是det(J)的绝对值）
            if det_J1 < 0:
                neg_det_count_J1 += 1
                # 静默处理：MATLAB代码中没有警告，直接使用绝对值
                det_J1 = abs(det_J1)
            
            # MATLAB: dR_dPhy = inv(J1)*dR_dPara;
            # 基函数对物理坐标的导数：∂R/∂(x,y) = inv(∂(x,y)/∂(u,v)) * ∂R/∂(u,v)
            dR_dPhy = np.linalg.inv(J1) @ dR_dPara  # 2xNen
            
            # 应变-位移矩阵B
            Be = np.zeros((3, 2 * num_ctrlpts))
            Be[0, :num_ctrlpts] = dR_dPhy[0, :]
            Be[1, num_ctrlpts:] = dR_dPhy[1, :]
            Be[2, :num_ctrlpts] = dR_dPhy[1, :]
            Be[2, num_ctrlpts:] = dR_dPhy[0, :]
            
            # 参数空间到父空间的映射
            # MATLAB: dPara_dPare(1,1) = (Ele_Knot_U(2)-Ele_Knot_U(1))/2;
            # MATLAB: dPara_dPare(2,2) = (Ele_Knot_V(2)-Ele_Knot_V(1))/2;
            dPara_dPare = np.zeros((2, 2))
            dPara_dPare[0, 0] = (Ele_Knot_U[1] - Ele_Knot_U[0]) / 2
            dPara_dPare[1, 1] = (Ele_Knot_V[1] - Ele_Knot_V[0]) / 2
            
            J2 = dPara_dPare
            # MATLAB: J = J1*J2;  (矩阵乘法)
            J = J1 @ J2  # 物理空间到父空间的映射
            
            # 权重因子
            # MATLAB: weight = GauPts.Weigh(i)*det(J);
            # 注意：雅可比矩阵的行列式应该是正的，如果为负说明坐标顺序有问题
            det_J = np.linalg.det(J)
            
            # MATLAB代码中直接使用det(J)，如果为负也只是数值为负
            # 在体积计算中，使用det(J)的绝对值是合理的
            # 静默处理：MATLAB代码中没有警告
            if det_J < 0:
                neg_det_count_J += 1
                det_J = abs(det_J)
            
            weight = GauPts['Weigh'][i] * det_J
            
            # 检查权重是否有效
            if weight <= 0 or np.isnan(weight) or np.isinf(weight):
                print(f"警告: 单元{ide}高斯点{i}权重无效: weight={weight}, det_J={det_J}")
                weight = 1e-10  # 设置一个很小的正值避免除零
            
            # 材料属性插值
            rho = X['GauPts'][GptOrder]
            E = Emin + (rho ** penal) * (1 - Emin)
            
            # 单元刚度矩阵
            Ke += E * weight * (Be.T @ DH @ Be)
            
            # 刚度矩阵对密度的导数
            dKe_i = (penal * (rho ** (penal - 1)) * (1 - Emin)) * weight * (Be.T @ DH @ Be)
            dKe.append(dKe_i)
            
            # 体积对密度的导数
            dv_dg[GptOrder] = weight
        
        KE.append(Ke)
        dKE.append(dKe)
    
    # 打印统计信息
    print('\n=== Stiff_Ele2D 统计信息 ===')
    total_gpts = Ele['Num'] * Ele['GauPtsNum']
    print(f'J1负行列式数量: {neg_det_count_J1} / {total_gpts} ({100*neg_det_count_J1/total_gpts:.2f}%)')
    print(f'J负行列式数量: {neg_det_count_J} / {total_gpts} ({100*neg_det_count_J/total_gpts:.2f}%)')
    print('===========================\n')
    
    return KE, dKE, dv_dg

