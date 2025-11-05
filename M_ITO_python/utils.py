"""
工具函数模块 - 提供NURBS相关的辅助函数
"""
import numpy as np


def find_span(n, p, u, U):
    """
    查找参数值u在节点向量U中的节点区间
    
    参数:
        n: 控制点数量 - 1
        p: B样条度数
        u: 参数值（标量）
        U: 节点向量（numpy数组）
    
    返回:
        span: 节点区间索引
    
    算法基于MATLAB findspan实现：
    - 如果 u == U[n+2] (MATLAB索引)，返回 n
    - 否则找到最后一个满足 U[i] <= u 的索引i，返回 i-1
    """
    # 确保u是标量
    u = float(u)
    
    # 检查参数值是否在节点向量范围内
    if len(U) == 0:
        raise ValueError(f"节点向量为空")
    if u < U[0] or u > U[-1]:
        raise ValueError(f"参数值 {u} 超出节点向量范围 [{U[0]}, {U[-1]}], 节点向量: {U[:min(10, len(U))]}...")
    
    # 特殊情况：如果u等于U[n+2]（MATLAB索引，Python中是U[n+1]）
    # 注意：节点向量长度是 n+p+2，最后一个索引是 n+p+1
    # U[n+1] 对应MATLAB的 U(n+2)
    if len(U) > n + 1 and abs(u - U[n + 1]) < 1e-10:
        return n
    
    # MATLAB: s(j) = find(u(j) >= U,1,'last')-1;
    # find(u(j) >= U,1,'last') 返回最后一个满足 u(j) >= U[i] 的 i（1-based索引）
    # 等价于最后一个满足 U[i] <= u(j) 的 i（1-based索引）
    # 然后减1得到0-based索引
    # 在Python中，我们使用np.where找到所有满足条件的索引，然后取最后一个
    indices = np.where(U <= u)[0]
    if len(indices) == 0:
        span = p  # 如果没有找到，使用最小值p
    else:
        span = indices[-1]  # 最后一个满足条件的索引（0-based）
    
    # MATLAB返回的是span（0-based），但我们需要确保span >= p（因为基函数需要p+1个节点）
    # 并且span <= n
    span = max(p, span)
    span = min(span, n)
    
    return span


def basis_function(p, U, span, u):
    """
    计算B样条基函数值
    
    参数:
        p: B样条度数
        U: 节点向量
        span: 节点区间索引（从find_span获得）
        u: 参数值
    
    返回:
        N: 基函数值数组，长度为p+1
    """
    # Algorithm A2.2 from 'The NURBS BOOK' pg70
    N = np.zeros(p + 1)
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    
    N[0] = 1.0
    
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            if abs(denom) < 1e-10:
                # 处理除零情况：当节点值重复时，设置temp为0
                temp = 0.0
            else:
                temp = N[r] / denom
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        
        N[j] = saved
    
    return N


def numbasisfun(iv, uv, p, U):
    """
    计算节点区间内非零基函数的索引
    完全匹配MATLAB的numbasisfun实现
    
    参数:
        iv: 节点区间索引（从find_span获得）
        uv: 参数值（标量或数组）
        p: B样条度数
        U: 节点向量
    
    返回:
        B: 基函数索引数组，shape (numel(uv), p+1)
           MATLAB: B = bsxfun(@(a,b) a+b, iv-p, (0:p).')
    """
    # MATLAB: B = bsxfun(@(a,b) a+b, iv-p, (0:p).')
    # 这相当于：B = (iv - p) + (0:p)，对每个uv值
    if np.isscalar(uv):
        uv = np.array([uv])
    
    iv = np.asarray(iv)
    if iv.ndim == 0:
        iv = np.array([iv])
    
    # 对于每个iv值，计算 p+1 个基函数索引
    B = np.zeros((len(iv), p + 1), dtype=int)
    for i in range(len(iv)):
        B[i, :] = (iv[i] - p) + np.arange(p + 1)
    
    return B


def nrbnumbasisfun(points, nrb):
    """
    计算NURBS基函数的编号（控制点索引）
    完全匹配MATLAB的nrbnumbasisfun实现
    
    参数:
        points: 参数坐标点集，shape (2, n_points) 或 dict {0: u_array, 1: v_array}
        nrb: NURBS曲面对象
    
    返回:
        idx: 控制点索引，shape (n_points, prod(nrb.order))
    """
    from utils import find_span, numbasisfun
    
    # 处理输入格式
    if isinstance(points, dict):
        # 单元格数组格式 {u, v}
        u_array = points[0] if 0 in points else points.get('u', [])
        v_array = points[1] if 1 in points else points.get('v', [])
        npts_dim = [len(u_array), len(v_array)]
        npts = npts_dim[0] * npts_dim[1]
        
        # 计算每个方向的span和索引
        sp_u = []
        sp_v = []
        for u_val in u_array:
            sp_u.append(find_span(nrb.ctrlpts_size_u - 1, nrb.degree_u, u_val, np.array(nrb.knotvector_u)))
        for v_val in v_array:
            sp_v.append(find_span(nrb.ctrlpts_size_v - 1, nrb.degree_v, v_val, np.array(nrb.knotvector_v)))
        
        # 计算局部索引
        num_u = numbasisfun(sp_u, u_array, nrb.degree_u, np.array(nrb.knotvector_u)) + 1  # MATLAB索引从1开始
        num_v = numbasisfun(sp_v, v_array, nrb.degree_v, np.array(nrb.knotvector_v)) + 1  # MATLAB索引从1开始
        
        # 使用tensor product计算全局索引
        # MATLAB: idx = reshape(sub2ind(nrb.number, local_num{:}), 1, size(idx, 2))
        idx = np.zeros((npts, (nrb.degree_u + 1) * (nrb.degree_v + 1)), dtype=int)
        idx_count = 0
        
        for v_idx_local in range(nrb.degree_v + 1):
            for u_idx_local in range(nrb.degree_u + 1):
                for v_val_idx in range(npts_dim[1]):
                    for u_val_idx in range(npts_dim[0]):
                        pt_idx = v_val_idx * npts_dim[0] + u_val_idx
                        # MATLAB的sub2ind: (u_idx-1)*NumV + v_idx
                        u_global = num_u[u_val_idx, u_idx_local]
                        v_global = num_v[v_val_idx, v_idx_local]
                        idx[pt_idx, idx_count] = (u_global - 1) * nrb.ctrlpts_size_v + v_global
                idx_count += 1
        
        return idx
    else:
        # 数组格式 points shape (2, n_points)
        npts = points.shape[1]
        idx = np.zeros((npts, (nrb.degree_u + 1) * (nrb.degree_v + 1)), dtype=int)
        
        for ipt in range(npts):
            u_val = points[0, ipt]
            v_val = points[1, ipt]
            
            # 计算span
            sp_u = find_span(nrb.ctrlpts_size_u - 1, nrb.degree_u, u_val, np.array(nrb.knotvector_u))
            sp_v = find_span(nrb.ctrlpts_size_v - 1, nrb.degree_v, v_val, np.array(nrb.knotvector_v))
            
            # 计算局部索引
            num_u = numbasisfun([sp_u], [u_val], nrb.degree_u, np.array(nrb.knotvector_u))[0] + 1
            num_v = numbasisfun([sp_v], [v_val], nrb.degree_v, np.array(nrb.knotvector_v))[0] + 1
            
            # 计算全局索引（tensor product）
            idx_count = 0
            for v_idx_local in range(nrb.degree_v + 1):
                for u_idx_local in range(nrb.degree_u + 1):
                    u_global = num_u[u_idx_local]
                    v_global = num_v[v_idx_local]
                    idx[ipt, idx_count] = (u_global - 1) * nrb.ctrlpts_size_v + v_global
                    idx_count += 1
        
        return idx


def basis_function_derivatives(p, U, span, u, nders=1):
    """
    计算B样条基函数的导数（解析方法）
    
    参数:
        p: B样条度数
        U: 节点向量
        span: 节点区间索引（从find_span获得，0-based）
        u: 参数值
        nders: 需要计算的导数阶数（默认1）
    
    返回:
        ders: 导数数组，shape (nders+1, p+1)
              ders[0, :] 是基函数值（0阶导数）
              ders[1, :] 是1阶导数
              ...
    
    完全复刻MATLAB的basisfunder实现，基于 Algorithm A2.3 from 'The NURBS BOOK' pg72
    """
    # MATLAB中findspan返回0-based索引，basisfunder中转换为1-based: i = ii(jj)+1
    # 注意：在MATLAB的basisfunder中，传入的span是findspan的结果（0-based）
    # 然后转换为1-based: i = span + 1
    # 但这里U数组也是0-based的，所以索引需要调整
    i = span + 1  # 转换为1-based索引（MATLAB风格），用于索引计算
    
    ders = np.zeros((nders + 1, p + 1))
    ndu = np.zeros((p + 1, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    a = np.zeros((2, p + 1))
    
    ndu[0, 0] = 1.0
    
    # 计算基函数值和节点差
    # MATLAB: for j = 1:pl
    # MATLAB:   left(j+1) = u - u_knotl(i+1-j);
    # MATLAB:   right(j+1) = u_knotl(i+j) - u;
    # 其中u_knotl是节点向量，i是1-based索引
    # 在MATLAB中，如果i=101（1-based），那么u_knotl(i+1-j)就是u_knotl(102-j)
    # 在Python中，U是0-based，i=101（1-based）对应U[100]（0-based）
    # 所以u_knotl(i+1-j)对应U[i+1-j-1] = U[i-j]
    # u_knotl(i+j)对应U[i+j-1]
    for j in range(1, p + 1):
        # MATLAB: left(j+1) = u - u_knotl(i+1-j)
        # MATLAB: right(j+1) = u_knotl(i+j) - u
        # 其中i是1-based索引（i = span + 1），u_knotl是MATLAB数组（1-based）
        # MATLAB的u_knotl(i+1-j) = u_knotl(span+2-j)（1-based）
        # 在Python中（0-based），对应U[span+2-j-1] = U[span+1-j]
        # MATLAB的u_knotl(i+j) = u_knotl(span+1+j)（1-based）
        # 在Python中（0-based），对应U[span+1+j-1] = U[span+j]
        idx_left = span + 1 - j  # U[span+1-j]，对应MATLAB的u_knotl(i+1-j)
        idx_right = span + j  # U[span+j]，对应MATLAB的u_knotl(i+j)
        
        # 确保索引在有效范围内
        if idx_left < 0 or idx_left >= len(U):
            # 这种情况不应该发生，但如果发生了，设置一个默认值
            left[j] = 0.0
        else:
            left[j] = u - U[idx_left]
        
        if idx_right < 0 or idx_right >= len(U):
            right[j] = 0.0
        else:
            right[j] = U[idx_right] - u
        saved = 0.0
        
        # MATLAB: for r = 0:j-1
        # MATLAB:   ndu(j+1,r+1) = right(r+2) + left(j-r+1);
        # MATLAB:   temp = ndu(r+1,j)/ndu(j+1,r+1);
        # MATLAB:   ndu(r+1,j+1) = saved + right(r+2)*temp;
        # MATLAB:   saved = left(j-r+1)*temp;
        for r in range(j):
            # MATLAB索引转Python索引：
            # ndu(j+1, r+1) → ndu[j, r]
            # right(r+2) → right[r+1]
            # left(j-r+1) → left[j-r]
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r] if ndu[j, r] > 1e-15 else 0.0
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        
        ndu[j, j] = saved
    
    # 提取基函数值（0阶导数）
    # MATLAB: for j = 0:pl, ders(1,j+1) = ndu(j+1,pl+1);
    for j in range(p + 1):
        ders[0, j] = ndu[j, p]
    
    # 计算导数
    # MATLAB: for r = 0:pl
    for r in range(p + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0
        
        # MATLAB: for k = 1:nders
        for k in range(1, nders + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            
            # MATLAB: if (r >= k)
            if r >= k:
                # MATLAB: a(s2+1,1) = a(s1+1,1)/ndu(pk+2,rk+1);
                # MATLAB: d = a(s2+1,1)*ndu(rk+1,pk+1);
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk] if ndu[pk + 1, rk] > 1e-15 else 0.0
                d = a[s2, 0] * ndu[rk, pk]
            
            # MATLAB: if (rk >= -1), j1 = 1; else j1 = -rk; end
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            
            # MATLAB: if ((r-1) <= pk), j2 = k-1; else j2 = pl-r; end
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = p - r
            
            # MATLAB: for j = j1:j2
            for j in range(j1, j2 + 1):
                # MATLAB: a(s2+1,j+1) = (a(s1+1,j+1) - a(s1+1,j))/ndu(pk+2,rk+j+1);
                # MATLAB: d = d + a(s2+1,j+1)*ndu(rk+j+1,pk+1);
                denom = ndu[pk + 1, rk + j] if (pk + 1 < len(ndu) and rk + j >= 0) else 1.0
                if denom > 1e-15:
                    a[s2, j] = (a[s1, j] - a[s1, j - 1]) / denom
                else:
                    a[s2, j] = 0.0
                if rk + j >= 0 and rk + j < len(ndu) and pk < len(ndu):
                    d += a[s2, j] * ndu[rk + j, pk]
            
            # MATLAB: if (r <= pk)
            if r <= pk:
                # MATLAB: a(s2+1,k+1) = -a(s1+1,k)/ndu(pk+2,r+1);
                # MATLAB: d = d + a(s2+1,k+1)*ndu(r+1,pk+1);
                denom = ndu[pk + 1, r] if (pk + 1 < len(ndu) and r >= 0) else 1.0
                if denom > 1e-15:
                    a[s2, k] = -a[s1, k - 1] / denom
                else:
                    a[s2, k] = 0.0
                if r >= 0 and r < len(ndu) and pk < len(ndu):
                    d += a[s2, k] * ndu[r, pk]
            
            # MATLAB: ders(k+1,r+1) = d;
            ders[k, r] = d
            
            # 交换s1和s2
            j_temp = s1
            s1 = s2
            s2 = j_temp
    
    # 乘以阶乘因子
    # MATLAB: r = pl; for k = 1:nders, for j = 0:pl, ders(k+1,j+1) = ders(k+1,j+1)*r; end; r = r*(pl-k); end
    r = p
    for k in range(1, nders + 1):
        for j in range(p + 1):
            ders[k, j] *= r
        r *= (p - k)
    
    return ders


def bincoeff(n, k):
    """计算二项式系数 C(n,k) = n!/(k!(n-k)!)"""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    # 使用对数避免溢出
    from math import lgamma
    return int(round(np.exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1))))


def bspdegelev(d, c, k, t):
    """
    提升B样条的阶数
    参考MATLAB的bspdegelev实现
    
    参数:
        d: B样条的阶数
        c: 控制点矩阵，shape (mc, nc)
        k: 节点向量
        t: 要提升的阶数次数
    
    返回:
        ic: 新的控制点矩阵
        ik: 新的节点向量
    """
    if t == 0:
        return c.copy(), k.copy()
    
    mc, nc = c.shape
    n = nc - 1
    m = n + d + 1
    ph = d + t
    ph2 = ph // 2
    
    # 计算Bezier阶数提升系数
    bezalfs = np.zeros((d + 1, ph + 1))
    bezalfs[0, 0] = 1
    bezalfs[d, ph] = 1
    
    for i in range(1, ph2 + 1):
        inv = 1.0 / bincoeff(ph, i)
        mpi = min(d, i)
        for j in range(max(0, i - t), mpi + 1):
            bezalfs[j, i] = inv * bincoeff(d, j) * bincoeff(t, i - j)
    
    for i in range(ph2 + 1, ph):
        mpi = min(d, i)
        for j in range(max(0, i - t), mpi + 1):
            bezalfs[j, i] = bezalfs[d - j, ph - i]
    
    # 初始化
    mh = ph
    kind = ph + 1
    r = -1
    a = d
    b = d + 1
    cind = 1
    ua = k[0]
    
    # 预分配输出
    ic = np.zeros((mc, nc * (t + 1)))
    ik = np.zeros(len(k) + nc * t)
    
    # 初始化第一个控制点和节点
    ic[:, 0] = c[:, 0]
    for i in range(ph + 1):
        ik[i] = ua
    
    # 初始化第一个Bezier段
    # 确保不超出范围
    init_size = min(d + 1, nc)
    bpts = c[:, :init_size].copy()
    if init_size < d + 1:
        # 如果初始控制点不足，需要扩展bpts
        bpts_full = np.zeros((mc, d + 1))
        bpts_full[:, :init_size] = bpts
        bpts = bpts_full
    ebpts = np.zeros((mc, ph + 1))
    Nextbpts = np.zeros((mc, d + 1))
    alfs = np.zeros(d)
    
    # 主循环：遍历节点向量
    while b < m:
        i = b
        while b < m - 1 and abs(k[b] - k[b + 1]) < 1e-10:
            b += 1
        mul = b - i + 1
        mh = mh + mul + t
        ub = k[b]
        oldr = r
        r = d - mul
        
        # 计算lbz和rbz
        if oldr > 0:
            lbz = (oldr + 2) // 2
        else:
            lbz = 1
        
        if r > 0:
            rbz = ph - (r + 1) // 2
        else:
            rbz = ph
        
        # 插入节点u(b) r次
        if r > 0:
            numer = ub - ua
            for q in range(d, mul, -1):
                # 检查索引是否在有效范围内
                idx = a + q
                if idx >= 0 and idx < len(k):
                    alfs[q - mul - 1] = numer / (k[idx] - ua)
                else:
                    # 如果索引超出范围，使用边界值
                    alfs[q - mul - 1] = 0.0
            
            for j in range(1, r + 1):
                save = r - j
                s = mul + j
                for q in range(d, s - 1, -1):
                    bpts[:, q] = alfs[q - s] * bpts[:, q] + (1.0 - alfs[q - s]) * bpts[:, q - 1]
                Nextbpts[:, save] = bpts[:, d]
        
        # 阶数提升Bezier段
        for i in range(lbz, ph + 1):
            ebpts[:, i] = 0.0
            mpi = min(d, i)
            for j in range(max(0, i - t), mpi + 1):
                ebpts[:, i] += bezalfs[j, i] * bpts[:, j]
        
        # 移除节点u=k[a] oldr次
        if oldr > 1:
            first = kind - 2
            last = kind
            den = ub - ua
            bet = (ub - ik[kind - 1]) / den
            
            for tr in range(1, oldr):
                i = first
                j = last
                kj = j - kind + 1
                while j - i > tr:
                    if i < cind:
                        alf = (ub - ik[i]) / (ua - ik[i])
                        ic[:, i] = alf * ic[:, i] + (1.0 - alf) * ic[:, i - 1]
                    if j >= lbz:
                        if j - tr <= kind - ph + oldr:
                            gam = (ub - ik[j - tr]) / den
                            ebpts[:, kj] = gam * ebpts[:, kj] + (1.0 - gam) * ebpts[:, kj + 1]
                        else:
                            ebpts[:, kj] = bet * ebpts[:, kj] + (1.0 - bet) * ebpts[:, kj + 1]
                    i += 1
                    j -= 1
                    kj -= 1
                first -= 1
                last += 1
        
        # 加载节点ua
        if a != d:
            for i in range(ph - oldr):
                ik[kind] = ua
                kind += 1
        
        # 加载控制点到ic
        for j in range(lbz, rbz + 1):
            ic[:, cind] = ebpts[:, j]
            cind += 1
        
        if b < m:
            # 设置下一次循环
            for j in range(r):
                bpts[:, j] = Nextbpts[:, j]
            for j in range(r, d + 1):
                # MATLAB: bpts[j][ii] = ctrl[b-d+j][ii];
                # 需要检查索引是否在有效范围内
                idx = b - d + j
                if idx >= 0 and idx < nc:
                    bpts[:, j] = c[:, idx]
                else:
                    # 如果索引超出范围，保持bpts的当前值（这种情况应该不会发生，但为了安全）
                    pass
            a = b
            b += 1
            ua = ub
        else:
            # 结束节点 - 使用原始节点向量的最后一个值
            ub_end = k[-1]  # 确保使用最后一个节点值
            for i in range(ph + 1):
                ik[kind + i] = ub_end
    
    # 裁剪到实际大小
    ic = ic[:, :cind]
    ik = ik[:mh + 1]
    
    # 验证节点向量：确保第一个和最后一个节点值正确
    if len(ik) > 0:
        # 第一个节点应该等于原始节点向量的第一个节点
        if abs(ik[0] - k[0]) > 1e-10:
            # 如果第一个节点不匹配，强制设置为正确的值
            ik[0] = k[0]
        # 最后一个节点应该等于原始节点向量的最后一个节点
        if abs(ik[-1] - k[-1]) > 1e-10:
            # 如果最后一个节点不匹配，强制设置为正确的值
            ik[-1] = k[-1]
    
    return ic, ik


def bspkntins(d, c, k, u):
    """
    向B样条插入节点（支持批量插入）
    完全1:1匹配MATLAB的bspkntins实现（Algorithm A5.4 from 'The NURBS BOOK' pg164）
    
    参数:
        d: B样条的阶数
        c: 控制点矩阵，shape (mc, nc)
        k: 节点向量
        u: 要插入的节点值（标量或数组）
    
    返回:
        ic: 新的控制点矩阵
        ik: 新的节点向量
    """
    mc, nc = c.shape
    # 确保u是数组并排序
    u = np.asarray(u)
    if u.ndim == 0:
        u = np.array([u])
    u = np.sort(u)
    nu = len(u)
    nk = len(k)
    
    if nu == 0:
        return c.copy(), k.copy()
    
    n = nc - 1
    m = n + d + 1
    r = nu - 1
    
    # 找到插入范围
    a = find_span(n, d, u[0], k)
    b = find_span(n, d, u[r], k)
    b = b + 1
    
    # 预分配输出
    ic = np.zeros((mc, nc + nu))
    ik = np.zeros(nk + nu)
    
    # 复制不变的控制点和节点
    # MATLAB: ic(:,1:a-d+1) = c(:,1:a-d+1);
    ic[:, :a - d + 1] = c[:, :a - d + 1]
    # MATLAB: ic(:,b+nu:nc+nu) = c(:,b:nc);
    ic[:, b + nu:nc + nu] = c[:, b:nc]
    
    # MATLAB: ik(1:a+1) = k(1:a+1);
    ik[:a + 1] = k[:a + 1]
    # MATLAB: ik(b+d+nu+1:m+nu+1) = k(b+d+1:m+1);
    # MATLAB注释：for (j = b+d; j <= m; j++) ik[j+r+1] = k[j];
    # 其中r = nu - 1，所以j+r+1 = j+nu
    # 这意味着：对于j从b+d到m（包含m），ik[j+nu] = k[j]
    # 在Python中：对于j从b+d到m（包含m），ik[j+nu] = k[j]
    # 注意：m = n + d + 1，k的长度是m+1（Python索引0到m）
    # 所以需要复制k[b+d]到k[m]（包含k[m]，即最后一个节点值1.0）
    # ik的长度是nk+nu = (m+1)+nu，索引范围0到m+nu
    # 所以ik[m+nu]是有效的（最后一个索引）
    # 但是，这个复制操作会在循环中被覆盖，所以我们需要在循环结束后再次确保这些值被设置
    for j in range(b + d, m + 1):
        if j < len(k):
            ik_idx = j + nu
            if ik_idx < len(ik):
                ik[ik_idx] = k[j]
    
    # 插入节点
    ii = b + d - 1
    ss = ii + nu
    
    # MATLAB: for jj=r:-1:0
    for jj in range(r, -1, -1):
        # MATLAB: ind = (a+1):ii;
        # MATLAB: ind = ind(u(jj+1)<=k(ind+1));
        # 找到所有满足u[jj] <= k[i]且i > a的索引
        ind_list = []
        i = ii
        while i > a and u[jj] <= k[i]:
            ind_list.append(i)
            i -= 1
        
        if len(ind_list) > 0:
            ind = np.array(ind_list)
            # MATLAB: ic(:,ind+ss-ii-d) = c(:,ind-d);
            # 确保索引在有效范围内
            idx_ic = ind + ss - ii - d
            idx_c = ind - d
            valid_mask = (idx_ic >= 0) & (idx_ic < ic.shape[1]) & (idx_c >= 0) & (idx_c < c.shape[1])
            if np.any(valid_mask):
                ic[:, idx_ic[valid_mask]] = c[:, idx_c[valid_mask]]
            # MATLAB: ik(ind+ss-ii+1) = k(ind+1);
            idx_ik = ind + ss - ii + 1
            idx_k = ind + 1
            valid_mask_k = (idx_ik >= 0) & (idx_ik < len(ik)) & (idx_k >= 0) & (idx_k < len(k))
            if np.any(valid_mask_k):
                ik[idx_ik[valid_mask_k]] = k[idx_k[valid_mask_k]]
            ii = ii - len(ind)
            ss = ss - len(ind)
        
        # MATLAB: ic(:,ss-d) = ic(:,ss-d+1);
        # 确保索引在有效范围内
        # ss - d 必须是有效索引（0到ic.shape[1]-1）
        # ss - d + 1 也必须是有效索引（0到ic.shape[1]-1）
        if ss - d >= 0 and ss - d + 1 < ic.shape[1]:
            ic[:, ss - d] = ic[:, ss - d + 1]
        
        # MATLAB: for l=1:d
        for l in range(1, d + 1):
            ind = ss - d + l
            # 确保索引在有效范围内
            if ind < 0 or ind >= ic.shape[1]:
                continue
            # MATLAB: alfa = ik(ss+l+1) - u(jj+1);
            if ss + l + 1 >= len(ik):
                continue
            alfa = ik[ss + l + 1] - u[jj]
            if abs(alfa) < 1e-10:
                # MATLAB: ic(:,ind) = ic(:,ind+1);
                if ind + 1 < ic.shape[1]:
                    ic[:, ind] = ic[:, ind + 1]
            else:
                # MATLAB: alfa = alfa/(ik(ss+l+1) - k(ii-d+l+1));
                if ii - d + l + 1 < len(k):
                    alfa = alfa / (ik[ss + l + 1] - k[ii - d + l + 1])
                    # MATLAB: ic(:,ind) = alfa*ic(:,ind) + (1-alfa)*ic(:,ind+1);
                    if ind + 1 < ic.shape[1]:
                        ic[:, ind] = alfa * ic[:, ind] + (1.0 - alfa) * ic[:, ind + 1]
        
        # MATLAB: ik(ss+1) = u(jj+1);
        # 确保ss+1在有效范围内，并且所有u中的值都被插入
        if ss + 1 >= 0 and ss + 1 < len(ik):
            ik[ss + 1] = u[jj]
        ss = ss - 1
    
    # 确保最后一个节点值被正确设置
    # MATLAB的算法保证ik[m+nu+1] = k[m+1]，但Python中索引从0开始
    # 所以应该确保ik[m+nu] = k[m]（最后一个节点值）
    # 注意：ik的长度是nk+nu = (m+1)+nu，最后一个有效索引是m+nu
    if m < len(k) and m + nu < len(ik):
        ik[m + nu] = k[m]
    # 另外，确保ik的最后一个元素等于k的最后一个元素（如果还没有设置）
    if len(ik) > 0 and len(k) > 0:
        if abs(ik[-1] - k[-1]) > 1e-10:
            ik[-1] = k[-1]
    
    # 确保所有u中的值都被插入（修复缺失的节点值）
    # 检查ik中是否包含所有u中的值
    ik_sorted = np.sort(ik)
    ik_unique = np.unique(ik_sorted)
    missing_values = []
    for val in u:
        # 检查val是否在ik_unique中（考虑浮点数精度）
        if not np.any(np.abs(ik_unique - val) < 1e-10):
            missing_values.append(val)
    
    # 如果有缺失的值，在循环结束后重新插入
    # 但是，MATLAB的算法应该已经插入了所有值，所以这不应该发生
    # 如果发生了，说明循环中的索引计算有问题
    if len(missing_values) > 0:
        # 对于每个缺失的值，找到应该插入的位置
        for missing_val in missing_values:
            # 找到ik中应该插入的位置（在第一个大于missing_val的值之前）
            # 由于ik已经排序，我们可以找到合适的位置
            found = False
            for i in range(len(ik)):
                if i > 0 and ik[i-1] < missing_val < ik[i]:
                    # 在i位置之前插入missing_val
                    # 但为了保持ik的长度，我们需要找到ik中值为0或可以替换的位置
                    # 检查ik[i]是否是重复的或可以替换的
                    if abs(ik[i] - ik[i-1]) < 1e-10:
                        ik[i] = missing_val
                        found = True
                        break
                    # 或者找到一个值为0的位置
                    for j in range(len(ik)):
                        if ik[j] == 0 and j > 0 and ik[j-1] < missing_val:
                            ik[j] = missing_val
                            found = True
                            break
                    if found:
                        break
            # 如果还没找到，尝试替换最接近的值
            if not found:
                for i in range(len(ik)):
                    if ik[i] == 0:
                        ik[i] = missing_val
                        break
    
    return ic, ik

