"""
几何建模模块 - 创建NURBS几何模型（不依赖geomdl）
直接实现MATLAB的Geom_Mod功能
"""
import numpy as np
from utils import bspdegelev, bspkntins, find_span, basis_function


class NURBSSurface:
    """简单的NURBS曲面类，不依赖geomdl"""
    def __init__(self):
        self.coefs = None  # 齐次坐标控制点，shape (4, num_u, num_v)
        self.knots = None  # 节点向量，dict {0: knots_u, 1: knots_v}
        self.degree_u = None
        self.degree_v = None
        self.ctrlpts_size_u = None
        self.ctrlpts_size_v = None
        self.knotvector_u = None
        self.knotvector_v = None
        self.ctrlpts = None  # 控制点列表（兼容性）
    
    def evaluate_single(self, uv):
        """评估单个参数点对应的齐次坐标 [xw, yw, zw, w]（仿 nrbeval 单点）"""
        u, v = uv
        num_u = self.ctrlpts_size_u
        num_v = self.ctrlpts_size_v
        p = self.degree_u
        q = self.degree_v
        U = np.array(self.knotvector_u)
        V = np.array(self.knotvector_v)

        # 查找区间
        span_u = find_span(num_u - 1, p, u, U)
        span_v = find_span(num_v - 1, q, v, V)

        # 基函数
        Nu = basis_function(p, U, span_u, u)
        Nv = basis_function(q, V, span_v, v)

        # 累加局部 (p+1)x(q+1) 控制点的齐次坐标
        xw = 0.0
        yw = 0.0
        zw = 0.0
        w = 0.0
        for j in range(q + 1):
            v_idx = span_v - q + j
            if v_idx < 0 or v_idx >= num_v:
                continue
            Nj = Nv[j]
            for i in range(p + 1):
                u_idx = span_u - p + i
                if u_idx < 0 or u_idx >= num_u:
                    continue
                Ni = Nu[i]
                B = Ni * Nj
                cw = self.coefs[:, u_idx, v_idx]  # [xw, yw, zw, w]
                xw += B * cw[0]
                yw += B * cw[1]
                zw += B * cw[2]
                w += B * cw[3]

        return [xw, yw, zw, w]


def geom_mod(L, W, Order, Num, BoundCon):
    """
    创建NURBS几何模型（不依赖geomdl）
    
    参数:
        L: 长度
        W: 宽度
        Order: B样条阶数增量 [p, q]（要提升的阶数）
        Num: 控制点数量 [n, m]
        BoundCon: 边界条件类型 (1-5)
    
    返回:
        NURBS: NURBS曲面对象
    """
    # 根据边界条件类型创建不同的几何模型
    if BoundCon in [1, 2, 3]:
        # 矩形域（直接按目标控制点数量构造规则网格 + 开区间均匀节点向量）
        # 最终度数：初始为1，再提升Order增量
        degree_u = 1 + int(Order[0])
        degree_v = 1 + int(Order[1])
        num_u = int(Num[0])
        num_v = int(Num[1])

        # 开区间均匀节点向量
        def open_uniform_knot(n, p):
            m = n + p + 1
            U = np.zeros(m)
            U[m - p - 1:] = 1.0
            for i in range(p + 1, n):  # i: p+1 .. n-1
                U[i] = (i - p) / (n - p)
            return U

        knots = {0: open_uniform_knot(num_u, degree_u), 1: open_uniform_knot(num_v, degree_v)}

        # 构造规则网格控制点（齐次坐标） - 列优先 (u 最快, v 最外)
        coefs = np.zeros((4, num_u, num_v))
        for v_idx in range(num_v):
            y = W * (v_idx / (num_v - 1)) if num_v > 1 else 0.0
            for u_idx in range(num_u):
                x = L * (u_idx / (num_u - 1)) if num_u > 1 else 0.0
                coefs[0, u_idx, v_idx] = x
                coefs[1, u_idx, v_idx] = y
                coefs[2, u_idx, v_idx] = 0.0
                coefs[3, u_idx, v_idx] = 1.0

    elif BoundCon == 4:
        # L型梁
        knots = {0: np.array([0, 0, 0.5, 1, 1]), 1: np.array([0, 0, 1, 1])}
        ControlPts = np.zeros((4, 3, 2))
        # MATLAB: ControlPts(:,:,1) = [0 0 L; L 0 0; 0 0 0; 1 1 1];
        ControlPts[:, :, 0] = np.array([[0, 0, L], [L, 0, 0], [0, 0, 0], [1, 1, 1]])  # (4, 3)
        # MATLAB: ControlPts(:,:,2) = [W W L; L W W; 0 0 0; 1 1 1];
        ControlPts[:, :, 1] = np.array([[W, W, L], [L, W, W], [0, 0, 0], [1, 1, 1]])  # (4, 3)
        
    elif BoundCon == 5:
        # 四分之一圆环
        W = W / 2
        knots = {0: np.array([0, 0, 0, 1, 1, 1]), 1: np.array([0, 0, 1, 1])}
        sqrt2_2 = np.sqrt(2) / 2
        ControlPts = np.zeros((4, 3, 2))
        # MATLAB: ControlPts(:,:,1) = [0 W W; W W 0; 0 0 0; 1 sqrt(2)/2 1];
        ControlPts[:, :, 0] = np.array([[0, W, W], [W, W, 0], [0, 0, 0], [1, sqrt2_2, 1]])  # (4, 3)
        # MATLAB: ControlPts(:,:,2) = [0 L L; L L 0; 0 0 0; 1 sqrt(2)/2 1];
        ControlPts[:, :, 1] = np.array([[0, L, L], [L, L, 0], [0, 0, 0], [1, sqrt2_2, 1]])  # (4, 3)
    
    # MATLAB: coefs = zeros(size(ControlPts));
    # 仅在 L 型梁与四分之一圆环分支中使用原始 ControlPts 组装
    # 矩形域已在上方直接构造 coefs
    if BoundCon in [4, 5]:
        coefs = np.zeros_like(ControlPts)
        coefs[0, :, :] = ControlPts[0, :, :] * ControlPts[3, :, :]
        coefs[1, :, :] = ControlPts[1, :, :] * ControlPts[3, :, :]
        coefs[2, :, :] = ControlPts[2, :, :] * ControlPts[3, :, :]
        coefs[3, :, :] = ControlPts[3, :, :]
    
    # MATLAB: NURBS = nrbmak(coefs, knots);
    # 计算初始阶数
    num_u = coefs.shape[1]
    num_v = coefs.shape[2]
    degree_u = len(knots[0]) - num_u - 1
    degree_v = len(knots[1]) - num_v - 1
    
    # 对于矩形域分支，已直接得到最终 degree/knots/coefs；
    # 对于其他分支（4、5），仍按原始流程进行升阶与插结
    if BoundCon in [4, 5]:
        # MATLAB: NURBS = nrbdegelev(NURBS,Order);
        # Order是要提升的阶数增量
        if Order[0] > 0 or Order[1] > 0:
            coefs, knots = nrbdegelev(coefs, knots, degree_u, degree_v, Order)
            # 更新阶数和控制点数量
            # 度数直接加上提升的阶数增量
            degree_u = degree_u + Order[0]
            degree_v = degree_v + Order[1]
            num_u = coefs.shape[1]
            num_v = coefs.shape[2]
    
    if BoundCon in [4, 5]:
        # MATLAB: iknot_u = linspace(0,1,Num(1)); iknot_v = linspace(0,1,Num(2));
        # NURBS = nrbkntins(NURBS,{setdiff(iknot_u,NURBS.knots{1}),setdiff(iknot_v,NURBS.knots{2})});
        iknot_u = np.linspace(0, 1, Num[0])
        iknot_v = np.linspace(0, 1, Num[1])
        
        # 获取唯一的节点值（排除已存在的节点）
        existing_u = np.unique(knots[0])
        existing_v = np.unique(knots[1])
        
        new_u = np.setdiff1d(iknot_u, existing_u)
        new_v = np.setdiff1d(iknot_v, existing_v)
        
        # 插入新节点
        if len(new_u) > 0 or len(new_v) > 0:
            coefs, knots = nrbkntins(coefs, knots, degree_u, degree_v, new_u, new_v)
    
    # 创建NURBS对象
    nrb = NURBSSurface()
    nrb.coefs = coefs
    nrb.knots = knots
    nrb.degree_u = degree_u
    nrb.degree_v = degree_v
    nrb.knotvector_u = knots[0].tolist()
    nrb.knotvector_v = knots[1].tolist()
    nrb.ctrlpts_size_u = coefs.shape[1]
    nrb.ctrlpts_size_v = coefs.shape[2]
    
    # 创建控制点列表（兼容性）
    # MATLAB按列优先存储，sub2ind([num_u, num_v], u, v) = (v-1)*num_u + u
    # 所以索引顺序应该是：(u=0,v=0), (u=1,v=0), ..., (u=num_u-1,v=0), (u=0,v=1), ...
    nrb.ctrlpts = []
    for v_idx in range(nrb.ctrlpts_size_v):
        for u_idx in range(nrb.ctrlpts_size_u):
            # 从齐次坐标转换为 [x, y, z, w] 格式
            w = coefs[3, u_idx, v_idx]
            if abs(w) > 1e-15:
                x = coefs[0, u_idx, v_idx] / w
                y = coefs[1, u_idx, v_idx] / w
                z = coefs[2, u_idx, v_idx] / w
            else:
                x = coefs[0, u_idx, v_idx]
                y = coefs[1, u_idx, v_idx]
                z = coefs[2, u_idx, v_idx]
            nrb.ctrlpts.append([x, y, z, w])
    
    return nrb


def nrbdegelev(coefs, knots, degree_u, degree_v, ntimes):
    """
    提升NURBS曲面的阶数
    参考MATLAB的nrbdegelev实现
    
    参数:
        coefs: 齐次坐标控制点，shape (4, num_u, num_v)
        knots: 节点向量字典 {0: knots_u, 1: knots_v}
        degree_u: u方向的阶数
        degree_v: v方向的阶数
        ntimes: 要提升的阶数增量 [ntimes_u, ntimes_v]
    
    返回:
        new_coefs: 新的控制点
        new_knots: 新的节点向量
    """
    dim, num_u, num_v = coefs.shape
    
    # 先提升v方向
    if ntimes[1] > 0:
        # 重塑为 (4*num_u, num_v)
        coefs_reshaped = coefs.reshape(4 * num_u, num_v)
        new_coefs_v, new_knots_v = bspdegelev(degree_v, coefs_reshaped, knots[1], ntimes[1])
        num_v = new_coefs_v.shape[1]
        coefs = new_coefs_v.reshape(4, num_u, num_v)
        knots_v = new_knots_v
    else:
        knots_v = knots[1]
    
    # 再提升u方向
    if ntimes[0] > 0:
        # 转置：permute(coefs, [1, 3, 2]) -> (4, num_v, num_u)
        coefs_permuted = np.transpose(coefs, (0, 2, 1))  # (4, num_v, num_u)
        # 重塑为 (4*num_v, num_u)
        coefs_reshaped = coefs_permuted.reshape(4 * num_v, num_u)
        new_coefs_u, new_knots_u = bspdegelev(degree_u, coefs_reshaped, knots[0], ntimes[0])
        num_u = new_coefs_u.shape[1]
        coefs_permuted = new_coefs_u.reshape(4, num_v, num_u)
        # 转置回来：permute(coefs, [1, 3, 2]) -> (4, num_u, num_v)
        coefs = np.transpose(coefs_permuted, (0, 2, 1))  # (4, num_u, num_v)
        knots_u = new_knots_u
    else:
        knots_u = knots[0]
    
    return coefs, {0: knots_u, 1: knots_v}


def nrbkntins(coefs, knots, degree_u, degree_v, new_u, new_v):
    """
    插入节点到NURBS曲面
    参考MATLAB的nrbkntins实现
    
    参数:
        coefs: 齐次坐标控制点，shape (4, num_u, num_v)
        knots: 节点向量字典 {0: knots_u, 1: knots_v}
        degree_u: u方向的阶数
        degree_v: v方向的阶数
        new_u: 要插入的u方向节点列表
        new_v: 要插入的v方向节点列表
    
    返回:
        new_coefs: 新的控制点
        new_knots: 新的节点向量
    """
    # MATLAB: bspkntins可以一次性插入多个节点
    # 先插入v方向节点
    if len(new_v) > 0:
        num_u = coefs.shape[1]
        num_v = coefs.shape[2]
        # 重塑为 (4*num_u, num_v)
        coefs_reshaped = coefs.reshape(4 * num_u, num_v)
        # MATLAB: [coefs,knots{2}] = bspkntins(degree(2),coefs,nurbs.knots{2},iknots{2});
        # 一次性插入所有v方向节点
        new_coefs_v, new_knots_v = bspkntins(degree_v, coefs_reshaped, knots[1], np.array(new_v))
        coefs_reshaped = new_coefs_v
        knots[1] = new_knots_v
        num_v = coefs_reshaped.shape[1]
        coefs = coefs_reshaped.reshape(4, num_u, num_v)
    
    # 再插入u方向节点
    if len(new_u) > 0:
        num_u = coefs.shape[1]
        num_v = coefs.shape[2]
        # MATLAB: coefs = permute(coefs,[1 3 2]);
        # 转置：permute(coefs, [1, 3, 2]) -> (4, num_v, num_u)
        coefs_permuted = np.transpose(coefs, (0, 2, 1))  # (4, num_v, num_u)
        # MATLAB: coefs = reshape(coefs,4*num2,num1);
        # 重塑为 (4*num_v, num_u)
        coefs_reshaped = coefs_permuted.reshape(4 * num_v, num_u)
        # MATLAB: [coefs,knots{1}] = bspkntins(degree(1),coefs,nurbs.knots{1},iknots{1});
        # 一次性插入所有u方向节点
        new_coefs_u, new_knots_u = bspkntins(degree_u, coefs_reshaped, knots[0], np.array(new_u))
        coefs_reshaped = new_coefs_u
        knots[0] = new_knots_u
        num_u = coefs_reshaped.shape[1]
        # MATLAB: coefs = reshape(coefs,[4 num2 size(coefs,2)]);
        coefs_permuted = coefs_reshaped.reshape(4, num_v, num_u)
        # MATLAB: coefs = permute(coefs,[1 3 2]);
        # 转置回来
        coefs = np.transpose(coefs_permuted, (0, 2, 1))  # (4, num_u, num_v)
    
    return coefs, knots
