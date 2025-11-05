"""
IGA预处理模块 - 提取控制点、单元和高斯点信息
完全1:1匹配MATLAB的Pre_IGA实现
"""
import numpy as np
from utils import find_span
from nrbbasisfun import nrbbasisfun


def pre_iga(NURBS):
    """
    IGA预处理
    完全1:1匹配MATLAB的Pre_IGA实现
    
    参数:
        NURBS: NURBS曲面对象
    
    返回:
        CtrPts: 控制点信息字典
        Ele: 单元信息字典
        GauPts: 高斯点信息字典
    """
    # MATLAB: Knots.U = unique(NURBS.knots{1})';
    # MATLAB: Knots.V = unique(NURBS.knots{2})';
    # 提取唯一节点
    knots_u = np.unique(NURBS.knotvector_u)
    knots_v = np.unique(NURBS.knotvector_v)
    
    # 控制点信息
    # MATLAB: CtrPts.Cordis = NURBS.coefs(:,:);
    # MATLAB的coefs是4 x (NumU*NumV)的矩阵，按列存储
    # Python中NURBS.coefs的shape是(4, num_u, num_v)
    # 需要按列优先展平成(4, num_u*num_v)
    num_u = NURBS.ctrlpts_size_u
    num_v = NURBS.ctrlpts_size_v
    
    # 关键修复：MATLAB的coefs(:,:)展平是列优先的
    # MATLAB: sub2ind([num_u, num_v], u, v) = u + (v-1)*num_u（列优先！）
    # coefs(4, :) 按列展平：coefs(:,1,1), coefs(:,2,1), ..., coefs(:,num_u,1), coefs(:,1,2), ...
    coefs_4d = np.zeros((4, num_u * num_v))
    for v_idx in range(num_v):
        for u_idx in range(num_u):
            # MATLAB: idx = u + (v-1)*num_u，Python（0-based）: idx = u_idx + v_idx*num_u
            idx = u_idx + v_idx * num_u
            coefs_4d[:, idx] = NURBS.coefs[:, u_idx, v_idx]
    
    # MATLAB: CtrPts.Cordis(1,:) = CtrPts.Cordis(1,:)./CtrPts.Cordis(4,:);
    # MATLAB: CtrPts.Cordis(2,:) = CtrPts.Cordis(2,:)./CtrPts.Cordis(4,:);
    # 转换为笛卡尔坐标（除以权重）
    cordis = np.zeros((2, num_u * num_v))
    for idx in range(num_u * num_v):
        w = coefs_4d[3, idx]
        if abs(w) > 1e-15:  # 避免除零
            cordis[0, idx] = coefs_4d[0, idx] / w  # x坐标
            cordis[1, idx] = coefs_4d[1, idx] / w  # y坐标
        else:
            cordis[0, idx] = coefs_4d[0, idx]
            cordis[1, idx] = coefs_4d[1, idx]
    
    CtrPts = {}
    CtrPts['Num'] = num_u * num_v
    CtrPts['NumU'] = num_u
    CtrPts['NumV'] = num_v
    CtrPts['Cordis'] = cordis  # 2 x Num 的矩阵，第一行是x坐标，第二行是y坐标
    
    # MATLAB: CtrPts.Seque = reshape(1:CtrPts.Num,CtrPts.NumU,CtrPts.NumV)';
    # MATLAB中'是转置，NumU是列数，NumV是行数
    # reshape后转置，所以最终是(NumV, NumU)形状
    CtrPts['Seque'] = np.arange(1, CtrPts['Num'] + 1).reshape(num_u, num_v, order='F').T
    
    # 单元信息
    # MATLAB: Ele.NumU = numel(unique(NURBS.knots{1}))-1;
    # MATLAB: Ele.NumV = numel(unique(NURBS.knots{2}))-1;
    Ele = {}
    Ele['NumU'] = len(knots_u) - 1
    Ele['NumV'] = len(knots_v) - 1
    Ele['Num'] = Ele['NumU'] * Ele['NumV']
    
    # MATLAB: Ele.Seque = reshape(1:Ele.Num, Ele.NumU, Ele.NumV)';
    Ele['Seque'] = np.arange(1, Ele['Num'] + 1).reshape(Ele['NumU'], Ele['NumV'], order='F').T
    
    # MATLAB: Ele.KnotsU = [Knots.U(1:end-1) Knots.U(2:end)];
    # MATLAB: Ele.KnotsV = [Knots.V(1:end-1) Knots.V(2:end)];
    Ele['KnotsU'] = np.column_stack([knots_u[:-1], knots_u[1:]])
    Ele['KnotsV'] = np.column_stack([knots_v[:-1], knots_v[1:]])
    
    # MATLAB: Ele.CtrPtsNum = prod(NURBS.order);
    # MATLAB: Ele.CtrPtsNumU = NURBS.order(1); Ele.CtrPtsNumV = NURBS.order(2);
    # 注意：MATLAB的order是度数+1，即order = [degree_u+1, degree_v+1]
    Ele['CtrPtsNum'] = (NURBS.degree_u + 1) * (NURBS.degree_v + 1)  # 每个单元的控制点数量
    Ele['CtrPtsNumU'] = NURBS.degree_u + 1
    Ele['CtrPtsNumV'] = NURBS.degree_v + 1
    
    # MATLAB: [~, Ele.CtrPtsCon] = nrbbasisfun({(sum(Ele.KnotsU,2)./2)', (sum(Ele.KnotsV,2)./2)'}, NURBS);
    # MATLAB中使用cell数组输入，计算每个单元中点的基函数
    # sum(Ele.KnotsU,2)是每行的和，./2是除以2，'是转置
    # 所以是计算每个单元的中点坐标
    u_mid = np.sum(Ele['KnotsU'], axis=1) / 2  # 每行的和除以2
    v_mid = np.sum(Ele['KnotsV'], axis=1) / 2  # 每行的和除以2
    
    # MATLAB的cell数组输入会生成tensor product网格
    # 对于每个(u_mid[i], v_mid[j])组合，计算基函数
    # MATLAB的tensor product顺序：kron(N{v}, N{u})意味着u方向变化最快
    # 所以遍历顺序：外层v，内层u
    Ele['CtrPtsCon'] = []
    
    for j in range(Ele['NumV']):
        for i in range(Ele['NumU']):
            # 使用nrbbasisfun计算该单元的控制点连接
            uv = np.array([[u_mid[i]], [v_mid[j]]])  # shape (2, 1)
            _, id_array = nrbbasisfun(uv, NURBS)
            
            # id_array的形状是(1, n_basis)，需要flatten
            Ele['CtrPtsCon'].append(id_array.flatten())
    
    # 转换为numpy数组
    Ele['CtrPtsCon'] = np.array(Ele['CtrPtsCon'])
    
    # 调试信息：打印Ele.CtrPtsCon的信息
    print('=== Pre_IGA Debug Info ===')
    print(f'Ele.Num = {Ele["Num"]}, Ele.CtrPtsNum = {Ele["CtrPtsNum"]}')
    print(f'Ele.CtrPtsCon size: [{Ele["CtrPtsCon"].shape[0]}, {Ele["CtrPtsCon"].shape[1]}]')
    if Ele['Num'] > 0:
        print(f'Ele.CtrPtsCon[0, :] = {Ele["CtrPtsCon"][0, :]}')
        if Ele['Num'] > 1:
            print(f'Ele.CtrPtsCon[1, :] = {Ele["CtrPtsCon"][1, :]}')
    # 调试信息：打印控制点坐标信息
    print(f'CtrPts.Cordis shape: {CtrPts["Cordis"].shape}')
    print(f'CtrPts.Cordis[0, 0:5] (x坐标) = {CtrPts["Cordis"][0, :min(5, CtrPts["Num"])]}')
    print(f'CtrPts.Cordis[1, 0:5] (y坐标) = {CtrPts["Cordis"][1, :min(5, CtrPts["Num"])]}')
    print(f'CtrPts.Cordis y坐标范围: [{CtrPts["Cordis"][1, :].min():.6f}, {CtrPts["Cordis"][1, :].max():.6f}]')
    print('==========================\n')
    
    # 高斯积分点信息
    # MATLAB: [GauPts.Weigh, GauPts.QuaPts] = Guadrature(3, numel(NURBS.order));
    from guadrature import guadrature
    GauPts_Weigh, GauPts_QuaPts = guadrature(3, 2)
    
    Ele['GauPtsNum'] = len(GauPts_Weigh)
    GauPts = {}
    GauPts['Num'] = Ele['Num'] * Ele['GauPtsNum']
    GauPts['Weigh'] = GauPts_Weigh
    GauPts['QuaPts'] = GauPts_QuaPts
    
    # MATLAB: GauPts.Seque = reshape(1:GauPts.Num,Ele.GauPtsNum,Ele.Num)';
    GauPts['Seque'] = np.arange(1, GauPts['Num'] + 1).reshape(Ele['GauPtsNum'], Ele['Num'], order='F').T
    
    GauPts['CorU'] = np.zeros((Ele['Num'], Ele['GauPtsNum']))
    GauPts['CorV'] = np.zeros((Ele['Num'], Ele['GauPtsNum']))
    
    for ide in range(Ele['Num']):
        # 找到单元在两个参数方向上的索引
        # MATLAB: [idv, idu] = find(Ele.Seque == ide);
        # Ele['Seque']的形状是(NumV, NumU)，索引从1开始
        idv, idu = np.where(Ele['Seque'] == ide + 1)
        if len(idu) > 0 and len(idv) > 0:
            idu = idu[0]
            idv = idv[0]
        else:
            # 如果找不到，手动计算索引
            idu = ide % Ele['NumU']
            idv = ide // Ele['NumU']
        
        Ele_Knot_U = Ele['KnotsU'][idu, :]
        Ele_Knot_V = Ele['KnotsV'][idv, :]
        
        # MATLAB: GauPts.CorU(ide,i) = ((Ele_Knot_U(2)-Ele_Knot_U(1)).*GauPts.QuaPts(i,1) + (Ele_Knot_U(2)+Ele_Knot_U(1)))/2;
        # MATLAB: GauPts.CorV(ide,i) = ((Ele_Knot_V(2)-Ele_Knot_V(1)).*GauPts.QuaPts(i,2) + (Ele_Knot_V(2)+Ele_Knot_V(1)))/2;
        for i in range(Ele['GauPtsNum']):
            GauPts['CorU'][ide, i] = ((Ele_Knot_U[1] - Ele_Knot_U[0]) * GauPts_QuaPts[i, 0] + 
                                      (Ele_Knot_U[1] + Ele_Knot_U[0])) / 2
            GauPts['CorV'][ide, i] = ((Ele_Knot_V[1] - Ele_Knot_V[0]) * GauPts_QuaPts[i, 1] + 
                                      (Ele_Knot_V[1] + Ele_Knot_V[0])) / 2
    
    return CtrPts, Ele, GauPts
