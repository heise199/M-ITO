"""
IGA 预处理 (IGA Preprocessing)
从 MATLAB 代码转换
"""

import numpy as np
from nurbs import nrbbasisfun
from guadrature import guadrature


def pre_iga(NURBS):
    """
    IGA 预处理 - 准备控制点、单元和高斯点信息
    
    参数:
        NURBS: NURBS 几何结构
    
    返回:
        CtrPts: 控制点信息字典
        Ele: 单元信息字典
        GauPts: 高斯积分点信息字典
    
    改编自 MATLAB IgaTop2D 代码
    """
    # ========== 节点信息 ==========
    Knots = {}
    Knots['U'] = np.unique(NURBS['knots'][0])
    Knots['V'] = np.unique(NURBS['knots'][1])
    
    # ========== 控制点信息 ==========
    CtrPts = {}
    
    # 确保 coefs 的形状正确：应该是 (4, NumU*NumV)
    if len(NURBS['coefs'].shape) == 3:
        # 如果是 (4, NumU, NumV)，需要重塑为 (4, NumU*NumV)
        # 使用 Fortran 顺序(flatten 按列)以与 MATLAB nrb 工具箱一致
        CtrPts['Cordis'] = NURBS['coefs'].reshape(4, -1, order='F').copy()
    else:
        CtrPts['Cordis'] = NURBS['coefs'].copy()
    
    # 转换为笛卡尔坐标（添加除零保护）
    weights = CtrPts['Cordis'][3, :]
    # 将接近零的权重设置为一个小值以避免除零
    weights = np.where(np.abs(weights) < 1e-10, 1.0, weights)
    CtrPts['Cordis'][0, :] = CtrPts['Cordis'][0, :] / weights
    CtrPts['Cordis'][1, :] = CtrPts['Cordis'][1, :] / weights
    CtrPts['Cordis'][2, :] = CtrPts['Cordis'][2, :] / weights
    CtrPts['Cordis'][3, :] = weights  # 更新权重
    
    CtrPts['Num'] = np.prod(NURBS['number'])  # 控制点总数
    CtrPts['NumU'] = NURBS['number'][0]  # U 方向控制点数
    CtrPts['NumV'] = NURBS['number'][1]  # V 方向控制点数
    # 注意：MATLAB reshape 是列优先（column-major），需要使用 order='F'
    CtrPts['Seque'] = np.arange(1, CtrPts['Num'] + 1).reshape(CtrPts['NumU'], CtrPts['NumV'], order='F').T
    
    # ========== 单元信息 ==========
    Ele = {}
    Ele['NumU'] = len(np.unique(NURBS['knots'][0])) - 1  # U 方向单元数
    Ele['NumV'] = len(np.unique(NURBS['knots'][1])) - 1  # V 方向单元数
    Ele['Num'] = Ele['NumU'] * Ele['NumV']  # 总单元数
    # MATLAB: reshape(1:N, NumU, NumV)'
    # 这相当于先按列填充(NumU, NumV)，然后转置得到(NumV, NumU)
    # 在Python中，使用order='F'先填充为列主序，然后转置
    Ele['Seque'] = np.arange(1, Ele['Num'] + 1).reshape(Ele['NumU'], Ele['NumV'], order='F').T
    
    Ele['KnotsU'] = np.column_stack([Knots['U'][:-1], Knots['U'][1:]])  # U 方向单元节点
    Ele['KnotsV'] = np.column_stack([Knots['V'][:-1], Knots['V'][1:]])  # V 方向单元节点
    
    Ele['CtrPtsNum'] = np.prod(NURBS['order'])  # 每个单元的控制点数
    Ele['CtrPtsNumU'] = NURBS['order'][0]
    Ele['CtrPtsNumV'] = NURBS['order'][1]
    
    # 计算单元-控制点连接
    # 在每个单元中心点求值基函数以获取非零基函数的索引
    # MATLAB: nrbbasisfun({(sum(Ele.KnotsU,2)./2)', (sum(Ele.KnotsV,2)./2)'}, NURBS)
    # 这里需要为每个单元找到其在网格中的位置，然后计算其中心点坐标
    centers_u = []
    centers_v = []
    for ide in range(Ele['Num']):
        # 找到单元在网格中的位置
        idv, idu = np.where(Ele['Seque'] == ide + 1)
        idv = idv[0]
        idu = idu[0]
        # 计算该单元的中心点坐标
        u_center = (Ele['KnotsU'][idu, 0] + Ele['KnotsU'][idu, 1]) / 2
        v_center = (Ele['KnotsV'][idv, 0] + Ele['KnotsV'][idv, 1]) / 2
        centers_u.append(u_center)
        centers_v.append(v_center)
    
    # 准备成点坐标形式 (2, Ele.Num)，每列是一个 (u, v) 点
    centers = np.vstack([centers_u, centers_v])
    
    _, Ele_CtrPtsCon = nrbbasisfun(centers, NURBS)
    Ele['CtrPtsCon'] = Ele_CtrPtsCon
    
    # 调试信息
    
    # ========== 高斯积分点信息 ==========
    GauPts = {}
    GauPts['Weigh'], GauPts['QuaPts'] = guadrature(3, len(NURBS['order']))
    Ele['GauPtsNum'] = len(GauPts['Weigh'])
    GauPts['Num'] = Ele['Num'] * Ele['GauPtsNum']
    GauPts['Seque'] = np.zeros((Ele['Num'], Ele['GauPtsNum']), dtype=int)
    for ide in range(Ele['Num']):
        start = ide * Ele['GauPtsNum']
        GauPts['Seque'][ide, :] = np.arange(start + 1, start + Ele['GauPtsNum'] + 1)
    
    GauPts['CorU'] = np.zeros((Ele['Num'], Ele['GauPtsNum']))
    GauPts['CorV'] = np.zeros((Ele['Num'], Ele['GauPtsNum']))
    
    for ide in range(Ele['Num']):
        # 使用 MATLAB 风格的 find(Ele.Seque == ide)
        # 在转置后的 Ele['Seque'] (NumV x NumU) 中查找单元 ide+1（1-based）
        idv, idu = np.where(Ele['Seque'] == ide + 1)
        idv = idv[0]  # 获取第一个（唯一）匹配
        idu = idu[0]
        Ele_Knot_U = Ele['KnotsU'][idu, :]
        Ele_Knot_V = Ele['KnotsV'][idv, :]
        
        for i in range(Ele['GauPtsNum']):
            # 从父空间映射到参数空间
            GauPts['CorU'][ide, i] = ((Ele_Knot_U[1] - Ele_Knot_U[0]) * GauPts['QuaPts'][i, 0] + 
                                       (Ele_Knot_U[1] + Ele_Knot_U[0])) / 2
            GauPts['CorV'][ide, i] = ((Ele_Knot_V[1] - Ele_Knot_V[0]) * GauPts['QuaPts'][i, 1] + 
                                       (Ele_Knot_V[1] + Ele_Knot_V[0])) / 2
    
    return CtrPts, Ele, GauPts


# ======================================================================================================================
# 函数: pre_iga
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

