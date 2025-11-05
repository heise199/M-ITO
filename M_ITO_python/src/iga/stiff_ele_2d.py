"""
单元刚度矩阵计算 - Stiff_Ele2D 的 Python 实现
2D平面应力问题（单位厚度假设）
"""
import numpy as np


def stiff_ele_2d(X, penal, Emin, DH, CtrPts, Ele, GauPts, dRu, dRv):
    """
    计算2D单元刚度矩阵（平面应力，单位厚度假设）
    与MATLAB的Stiff_Ele2D一致
    
    参数:
        X: 设计变量字典
        penal: SIMP惩罚因子
        Emin: 最小弹性模量
        DH: 本构矩阵 (3×3 for 2D平面应力)
        CtrPts: 控制点信息
        Ele: 元素信息
        GauPts: 高斯积分点信息
        dRu, dRv: 基函数对参数坐标u和v的导数
    
    返回:
        KE: 单元刚度矩阵列表 [Ke1, Ke2, ...]
        dKE: 单元刚度矩阵对密度的导数列表 [[dKe1_gpt1, ...], ...]
        dv_dg: 体积对密度的导数
    """
    KE = [None] * Ele['Num']
    dKE = [None] * Ele['Num']
    dv_dg = np.zeros(GauPts['Num'])
    
    Nen = Ele['CtrPtsNum']  # 每个元素的控制点数
    
    for ide in range(Ele['Num']):
        # 找到元素在两个方向的索引（2D：只使用u和v）
        # 2D：使用 (NumU, NumV) 而不是 (NumU, NumV, NumW)
        idx_u, idx_v = np.unravel_index(ide, (Ele['NumU'], Ele['NumV']))
        
        Ele_Knot_U = Ele['KnotsU'][idx_u, :]
        Ele_Knot_V = Ele['KnotsV'][idx_v, :]
        
        # 完全按照MATLAB版本：直接使用Ele.CtrPtsCon，不过滤
        # MATLAB: Ele_NoCtPt = Ele.CtrPtsCon(ide,:); （1-based索引）
        Ele_NoCtPt_raw = Ele['CtrPtsCon'][ide, :]  # 1-based MATLAB索引
        # 过滤掉0值（未初始化的控制点索引）
        valid_mask = Ele_NoCtPt_raw > 0
        valid_count = np.sum(valid_mask)
        if valid_count == 0:
            import warnings
            warnings.warn(f"元素 {ide} 没有有效的控制点索引！Ele['CtrPtsCon'][{ide}, :] = {Ele_NoCtPt_raw}")
            KE[ide] = np.zeros((2 * Ele['CtrPtsNum'], 2 * Ele['CtrPtsNum']))
            dKE[ide] = [None] * Ele['GauPtsNum']
            continue
        
        # 转换为0-based索引，只保留有效值（向量化）
        Ele_NoCtPt = Ele_NoCtPt_raw[valid_mask] - 1  # 0-based索引
        
        # 2D：只使用x和y坐标（Z坐标设为0）
        # MATLAB: Ele_CoCtPt = CtrPts.Cordis(1:2,Ele_NoCtPt);
        Ele_CoCtPt = CtrPts['Cordis'][0:2, Ele_NoCtPt]  # 控制点坐标 (2, Nen_valid)
        
        # 单元刚度矩阵大小：必须与Ele.CtrPtsNum匹配（MATLAB版本总是固定大小）
        # MATLAB: Ke = zeros(2*Nen,2*Nen);
        Nen = Ele['CtrPtsNum']  # 固定大小
        Nen_valid = len(Ele_NoCtPt)  # 有效控制点数量
        Ke = np.zeros((2 * Nen, 2 * Nen))  # 固定大小，与MATLAB一致
        dKe = [None] * Ele['GauPtsNum']
        
        for i in range(Ele['GauPtsNum']):
            GptOrder = GauPts['Seque'][ide, i] - 1  # Python索引
            
            # 基函数对参数坐标的导数
            # MATLAB: dR_dPara = [dRu(GptOrder,:); dRv(GptOrder,:)];
            # MATLAB中dR_dPara包含所有基函数的导数 (2, n_basis)
            # 然后矩阵乘法 dR_dPara*Ele_CoCtPt' 会自动选择Ele_NoCtPt对应的列
            # 
            # Python实现：直接选择对应的列，避免创建完整数组（优化）
            # 这样可以确保与MATLAB完全一致，同时减少内存分配
            dR_dPara = np.array([
                dRu[GptOrder, Ele_NoCtPt],  # 直接选择有效列
                dRv[GptOrder, Ele_NoCtPt]
            ])  # 形状: (2, Nen_valid)
            
            # 雅可比矩阵：从参数空间到物理空间 (2×2)
            # MATLAB: dPhy_dPara = dR_dPara*Ele_CoCtPt';
            # 等价于：选择dR_dPara中Ele_NoCtPt对应的列，然后与Ele_CoCtPt的转置相乘
            # Python: (2, Nen_valid) @ (Nen_valid, 2) = (2, 2)
            if Nen_valid < 2:
                import warnings
                warnings.warn(f"元素 {ide} 的有效控制点数 ({Nen_valid}) 少于2，无法计算雅可比矩阵")
                continue
            dPhy_dPara = dR_dPara @ Ele_CoCtPt.T
            J1 = dPhy_dPara
            
            # 检查雅可比矩阵是否奇异
            det_J1 = np.linalg.det(J1)
            if abs(det_J1) < 1e-10:
                import warnings
                warnings.warn(f"元素 {ide} 高斯点 {i} 的雅可比矩阵接近奇异！det(J1)={det_J1:.2e}")
            
            # 基函数对物理坐标的导数
            # MATLAB: dR_dPhy = inv(J1)*dR_dPara;
            # 使用solve代替inv更高效（MATLAB的inv实际上内部也可能优化）
            # 对于2x2矩阵，直接求逆也很快，但solve更稳定
            try:
                # 使用solve求解线性方程组：J1 @ dR_dPhy = dR_dPara
                # 等价于 dR_dPhy = inv(J1) @ dR_dPara，但更稳定
                dR_dPhy = np.linalg.solve(J1, dR_dPara)  # (2, 2) @ (2, Nen_valid) = (2, Nen_valid)
            except np.linalg.LinAlgError:
                # 如果奇异，使用伪逆（MATLAB的inv在奇异时会报错，这里使用伪逆作为备选）
                J1_inv = np.linalg.pinv(J1)
                dR_dPhy = J1_inv @ dR_dPara
            
            # 构建应变-位移矩阵 B (3×2Nen for 2D平面应力)
            # MATLAB: Be(1,1:Nen) = dR_dPhy(1,:); Be(2,Nen+1:2*Nen) = dR_dPhy(2,:);
            #         Be(3,1:Nen) = dR_dPhy(2,:); Be(3,Nen+1:2*Nen) = dR_dPhy(1,:);
            # 注意：MATLAB版本中Be的大小是(3, 2*Nen)，但只填充有效控制点的部分
            # Python版本中，我们需要创建(3, 2*Nen)大小的矩阵，但只填充有效部分
            # 向量化构建Be矩阵（避免循环）
            Be = np.zeros((3, 2 * Nen))
            # 只填充有效控制点的部分（向量化赋值）
            valid_slice = slice(0, Nen_valid)
            Be[0, valid_slice] = dR_dPhy[0, :]              # εxx = du/dx
            Be[1, Nen + valid_slice] = dR_dPhy[1, :]        # εyy = dv/dy
            Be[2, valid_slice] = dR_dPhy[1, :]               # γxy = du/dy + dv/dx
            Be[2, Nen + valid_slice] = dR_dPhy[0, :]
            
            # 从参数空间到父空间的映射（2D）
            dPara_dPare = np.zeros((2, 2))
            dPara_dPare[0, 0] = (Ele_Knot_U[1] - Ele_Knot_U[0]) / 2
            dPara_dPare[1, 1] = (Ele_Knot_V[1] - Ele_Knot_V[0]) / 2
            
            # 总雅可比：从物理空间到父空间 (2×2)
            J = J1 @ dPara_dPare
            # MATLAB: weight = GauPts.Weigh(i)*det(J);
            # 注意：MATLAB的det在2D情况下对于正常雅可比矩阵总是返回正值
            # 但为了数值稳定性，使用abs确保非负
            det_J = np.linalg.det(J)
            weight = GauPts['Weigh'][i] * abs(det_J)
            
            # 弹性模量（SIMP模型）
            E_gpt = Emin + X['GauPts'][GptOrder] ** penal * (1 - Emin)
            
            # 单元刚度矩阵
            Ke = Ke + E_gpt * weight * (Be.T @ DH @ Be)
            
            # 对密度的导数
            dKe[i] = (penal * X['GauPts'][GptOrder] ** (penal - 1) * (1 - Emin)) * weight * (Be.T @ DH @ Be)
            
            # 体积对密度的导数
            dv_dg[GptOrder] = weight
        
        KE[ide] = Ke
        dKE[ide] = dKe
    
    return KE, dKE, dv_dg

