"""
NURBS 操作函数 (节点插入、次数提升等)
从 MATLAB NURBS Toolbox 转换
"""

import numpy as np
from scipy.special import gammaln
from .bspline import findspan
from .nrb_core import nrbmak


def bincoeff(n, k):
    """
    计算二项式系数
    
    ( n )      n!
    (   ) = --------
    ( k )   k!(n-k)!
    
    改编自 'Numerical Recipes in C, 2nd Edition' pg215
    """
    return int(np.floor(0.5 + np.exp(gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1))))


def bspkntins(d, c, k, u):
    """
    向 B-Spline 插入节点
    
    参数:
        d: 样条次数 (整数)
        c: 控制点 (double 矩阵 (mc,nc))
        k: 节点序列 (double 向量 (nk))
        u: 新节点 (double 向量 (nu))
    
    返回:
        ic: 新控制点 (double 矩阵 (mc,nc+nu))
        ik: 新节点序列 (double 向量 (nk+nu))
    
    改编自 'The NURBS BOOK' pg164 Algorithm A5.4
    """
    mc, nc = c.shape
    u = np.sort(u)
    nu = len(u)
    nk = len(k)
    
    ic = np.zeros((mc, nc+nu))
    ik = np.zeros(nk+nu)
    
    n = nc - 1
    r = nu - 1
    
    m = n + d + 1
    a = int(findspan(n, d, u[0], k))
    b = int(findspan(n, d, u[r], k))
    b = b + 1
    
    # 复制未受影响的控制点
    ic[:, 0:a-d+1] = c[:, 0:a-d+1]
    ic[:, b+nu:nc+nu] = c[:, b:nc]
    
    # 复制节点
    ik[0:a+1] = k[0:a+1]
    ik[b+d+nu:m+nu+1] = k[b+d:m+1]
    
    ii = b + d - 1
    ss = ii + nu
    
    for jj in range(r, -1, -1):
        # MATLAB: ind = (a+1):ii; ind = ind(u(jj+1)<=k(ind+1));
        # Python: ind 从 a 到 ii-1，然后筛选满足 u[jj] <= k[ind+1] 的
        ind = np.arange(a, ii+1)  # 包含 ii
        if len(ind) > 0 and max(ind) + 1 < len(k):
            mask = u[jj] <= k[ind+1]
            ind = ind[mask]
        else:
            ind = np.array([], dtype=int)
        
        # 复制控制点和节点
        for i in ind:
            idx_ic = i + ss - ii - d
            idx_c = i - d
            if 0 <= idx_ic < ic.shape[1] and 0 <= idx_c < c.shape[1]:
                ic[:, idx_ic] = c[:, idx_c]
            idx_ik = i + ss - ii
            if 0 <= idx_ik < len(ik) and i + 1 < len(k):
                ik[idx_ik] = k[i+1]
        
        ii = ii - len(ind)
        ss = ss - len(ind)
        
        # ic(:,ss-d) = ic(:,ss-d+1);
        if 0 <= ss-d < ic.shape[1] and 0 <= ss-d+1 < ic.shape[1]:
            ic[:, ss-d] = ic[:, ss-d+1]
        
        for l in range(1, d+1):
            ind = ss - d + l
            if ind < 0 or ind >= ic.shape[1]:
                continue
            
            # alfa = ik(ss+l+1) - u(jj+1);
            if ss+l < len(ik):
                alfa = ik[ss+l] - u[jj]
            else:
                continue
                
            if abs(alfa) < 1e-14:
                # ic(:,ind) = ic(:,ind+1);
                if ind+1 < ic.shape[1]:
                    ic[:, ind] = ic[:, ind+1]
            else:
                # alfa = alfa/(ik(ss+l+1) - k(ii-d+l+1));
                idx_k = ii - d + l
                if 0 <= idx_k < len(k) and ss+l < len(ik):
                    denom = ik[ss+l] - k[idx_k]
                    if abs(denom) > 1e-14:
                        alfa = alfa / denom
                        if ind+1 < ic.shape[1]:
                            ic[:, ind] = alfa * ic[:, ind] + (1-alfa) * ic[:, ind+1]
        
        # ik(ss+1) = u(jj+1);
        if 0 <= ss < len(ik):
            ik[ss] = u[jj]
        ss = ss - 1
    
    return ic, ik


def bspdegelev(d, c, k, t):
    """
    提升单变量 B-Spline 的次数
    
    参数:
        d: B-Spline 的次数
        c: 控制点，大小为 (dim, nc) 的矩阵
        k: 节点序列，大小为 nk 的行向量
        t: 提升 B-Spline 次数 t 次
    
    返回:
        ic: 新 B-Spline 的控制点
        ik: 新 B-Spline 的节点向量
    """
    mc, nc = c.shape
    
    n = nc - 1
    
    bezalfs = np.zeros((d+1, d+t+1))
    bpts = np.zeros((mc, d+1))
    ebpts = np.zeros((mc, d+t+1))
    Nextbpts = np.zeros((mc, d+1))
    alfs = np.zeros(d)
    
    m = n + d + 1
    ph = d + t
    ph2 = ph // 2
    
    # 计算 Bezier 次数提升系数
    bezalfs[0, 0] = 1.0
    bezalfs[d, ph] = 1.0
    
    for i in range(1, ph2+1):
        inv = 1.0 / bincoeff(ph, i)
        mpi = min(d, i)
        
        for j in range(max(0, i-t), mpi+1):
            bezalfs[j, i] = inv * bincoeff(d, j) * bincoeff(t, i-j)
    
    for i in range(ph2+1, ph):
        mpi = min(d, i)
        for j in range(max(0, i-t), mpi+1):
            bezalfs[j, i] = bezalfs[d-j, ph-i]
    
    mh = ph
    kind = ph
    r = -1
    a = d
    b = d + 1
    cind = 0
    ua = k[0]
    
    ic = np.zeros((mc, n*(t+1) + 1 + t))  # 确保足够大
    for ii in range(mc):
        ic[ii, 0] = c[ii, 0]
    
    ik = np.zeros(m + n*t + 1 + t)  # 确保足够大
    for i in range(ph+1):
        ik[i] = ua
    
    # 初始化第一个 Bezier 段
    for i in range(d+1):
        for ii in range(mc):
            bpts[ii, i] = c[ii, i]
    
    # 主循环遍历节点向量
    while b < m:
        i = b
        while b < m and k[b] == k[b+1]:
            b = b + 1
        
        mul = b - i + 1
        mh = mh + mul + t
        ub = k[b]
        oldr = r
        r = d - mul
        
        # 插入节点 u(b) r 次
        if oldr > 0:
            lbz = (oldr + 2) // 2
        else:
            lbz = 1
        
        if r > 0:
            rbz = ph - (r + 1) // 2
        else:
            rbz = ph
        
        if r > 0:
            # 插入节点以获得 bezier 段
            numer = ub - ua
            for q in range(d, mul, -1):
                alfs[q-mul-1] = numer / (k[a+q] - ua)
            
            for j in range(1, r+1):
                save = r - j
                s = mul + j
                
                for q in range(d, s-1, -1):
                    for ii in range(mc):
                        bpts[ii, q] = alfs[q-s] * bpts[ii, q] + (1.0 - alfs[q-s]) * bpts[ii, q-1]
                
                for ii in range(mc):
                    Nextbpts[ii, save] = bpts[ii, d]
        
        # 提升 bezier 段次数
        for i in range(lbz, ph+1):
            for ii in range(mc):
                ebpts[ii, i] = 0.0
            mpi = min(d, i)
            for j in range(max(0, i-t), mpi+1):
                for ii in range(mc):
                    ebpts[ii, i] = ebpts[ii, i] + bezalfs[j, i] * bpts[ii, j]
        
        if oldr > 1:
            # 必须移除节点 u=k[a] oldr 次
            first = kind - 2
            last = kind
            den = ub - ua
            bet = (ub - ik[kind-1]) / den
            
            # 节点移除循环
            for tr in range(1, oldr):
                i = first
                j = last
                kj = j - kind + 1
                while j - i > tr:
                    if i < cind:
                        alf = (ub - ik[i]) / (ua - ik[i])
                        for ii in range(mc):
                            ic[ii, i] = alf * ic[ii, i] + (1.0 - alf) * ic[ii, i-1]
                    
                    if j >= lbz:
                        if j - tr <= kind - ph + oldr:
                            gam = (ub - ik[j-tr]) / den
                            for ii in range(mc):
                                ebpts[ii, kj] = gam * ebpts[ii, kj] + (1.0 - gam) * ebpts[ii, kj+1]
                        else:
                            for ii in range(mc):
                                ebpts[ii, kj] = bet * ebpts[ii, kj] + (1.0 - bet) * ebpts[ii, kj+1]
                    
                    i = i + 1
                    j = j - 1
                    kj = kj - 1
                
                first = first - 1
                last = last + 1
        
        # 加载节点 ua
        if a != d:
            for i in range(ph - oldr):
                ik[kind] = ua
                kind = kind + 1
        
        # 加载控制点到 ic
        for j in range(lbz, rbz+1):
            for ii in range(mc):
                ic[ii, cind] = ebpts[ii, j]
            cind = cind + 1
        
        if b < m:
            # 为下一次循环设置
            for j in range(r):
                for ii in range(mc):
                    bpts[ii, j] = Nextbpts[ii, j]
            
            for j in range(r, d+1):
                for ii in range(mc):
                    bpts[ii, j] = c[ii, b-d+j]
            
            a = b
            b = b + 1
            ua = ub
        else:
            # 结束节点
            for i in range(ph+1):
                ik[kind+i] = ub
    
    # 截断到实际使用的大小
    nh = mh - ph
    ic = ic[:, :nh]
    ik = ik[:mh+1]
    
    return ic, ik


def nrbkntins(nurbs, iknots):
    """
    向 NURBS 曲线、曲面或体插入节点
    
    参数:
        nurbs: NURBS 结构
        iknots: 要插入的节点
                对于曲线: 向量
                对于曲面: [iuknots, ivknots] 列表
                对于体: [iuknots, ivknots, iwknots] 列表
    
    返回:
        inurbs: 插入节点后的新 NURBS 结构
    """
    degree = [o-1 for o in nurbs['order']] if isinstance(nurbs['order'], list) else nurbs['order']-1
    
    if not isinstance(nurbs['knots'], list):  # 曲线
        if len(iknots) > 0 and (np.any(np.array(iknots) > np.max(nurbs['knots'])) or 
                                 np.any(np.array(iknots) < np.min(nurbs['knots']))):
            raise ValueError('尝试在定义区间之外插入节点')
        
        if len(iknots) == 0:
            coefs = nurbs['coefs']
            knots = nurbs['knots']
        else:
            coefs, knots = bspkntins(degree, nurbs['coefs'], nurbs['knots'], iknots)
    
    elif len(nurbs['knots']) == 2:  # 曲面
        num1 = nurbs['number'][0]
        num2 = nurbs['number'][1]
        
        # 检查节点范围
        for idim in range(2):
            if len(iknots[idim]) > 0:
                if (np.any(np.array(iknots[idim]) > np.max(nurbs['knots'][idim])) or 
                    np.any(np.array(iknots[idim]) < np.min(nurbs['knots'][idim]))):
                    raise ValueError('尝试在定义区间之外插入节点')
        
        # 沿 v 方向插入节点
        if len(iknots[1]) == 0:
            coefs = nurbs['coefs']
            knots2 = nurbs['knots'][1]
        else:
            coefs = nurbs['coefs'].reshape(4*num1, num2)
            coefs, knots2 = bspkntins(degree[1], coefs, nurbs['knots'][1], iknots[1])
            num2 = coefs.shape[1]
            coefs = coefs.reshape(4, num1, num2)
        
        # 沿 u 方向插入节点
        if len(iknots[0]) == 0:
            knots1 = nurbs['knots'][0]
        else:
            coefs = np.transpose(coefs, (0, 2, 1))
            coefs = coefs.reshape(4*num2, num1)
            coefs, knots1 = bspkntins(degree[0], coefs, nurbs['knots'][0], iknots[0])
            coefs = coefs.reshape(4, num2, coefs.shape[1])
            coefs = np.transpose(coefs, (0, 2, 1))
        
        knots = [knots1, knots2]
    
    elif len(nurbs['knots']) == 3:  # 体
        num1 = nurbs['number'][0]
        num2 = nurbs['number'][1]
        num3 = nurbs['number'][2]
        
        # 检查节点范围
        for idim in range(3):
            if len(iknots[idim]) > 0:
                if (np.any(np.array(iknots[idim]) > np.max(nurbs['knots'][idim])) or 
                    np.any(np.array(iknots[idim]) < np.min(nurbs['knots'][idim]))):
                    raise ValueError('尝试在定义区间之外插入节点')
        
        # 沿 w 方向插入节点
        if len(iknots[2]) == 0:
            coefs = nurbs['coefs']
            knots3 = nurbs['knots'][2]
        else:
            coefs = nurbs['coefs'].reshape(4*num1*num2, num3)
            coefs, knots3 = bspkntins(degree[2], coefs, nurbs['knots'][2], iknots[2])
            num3 = coefs.shape[1]
            coefs = coefs.reshape(4, num1, num2, num3)
        
        # 沿 v 方向插入节点
        if len(iknots[1]) == 0:
            knots2 = nurbs['knots'][1]
        else:
            coefs = np.transpose(coefs, (0, 1, 3, 2))
            coefs = coefs.reshape(4*num1*num3, num2)
            coefs, knots2 = bspkntins(degree[1], coefs, nurbs['knots'][1], iknots[1])
            num2 = coefs.shape[1]
            coefs = coefs.reshape(4, num1, num3, num2)
            coefs = np.transpose(coefs, (0, 1, 3, 2))
        
        # 沿 u 方向插入节点
        if len(iknots[0]) == 0:
            knots1 = nurbs['knots'][0]
        else:
            coefs = np.transpose(coefs, (0, 2, 3, 1))
            coefs = coefs.reshape(4*num2*num3, num1)
            coefs, knots1 = bspkntins(degree[0], coefs, nurbs['knots'][0], iknots[0])
            coefs = coefs.reshape(4, num2, num3, coefs.shape[1])
            coefs = np.transpose(coefs, (0, 3, 1, 2))
        
        knots = [knots1, knots2, knots3]
    
    else:
        raise ValueError('不支持的 NURBS 维度')
    
    # 构造新的 NURBS
    inurbs = nrbmak(coefs, knots)
    
    return inurbs


def nrbdegelev(nurbs, ntimes):
    """
    提升 NURBS 曲线、曲面或体的次数
    
    参数:
        nurbs: NURBS 结构
        ntimes: 提升次数
                对于曲线: 标量
                对于曲面: [utimes, vtimes]
                对于体: [utimes, vtimes, wtimes]
    
    返回:
        inurbs: 次数提升后的新 NURBS 结构
    """
    degree = [o-1 for o in nurbs['order']] if isinstance(nurbs['order'], list) else nurbs['order']-1
    
    if not isinstance(nurbs['knots'], list):  # 曲线
        if ntimes == 0 or ntimes is None:
            coefs = nurbs['coefs']
            knots = nurbs['knots']
        else:
            coefs, knots = bspdegelev(degree, nurbs['coefs'], nurbs['knots'], ntimes)
    
    elif len(nurbs['knots']) == 2:  # 曲面
        dim, num1, num2 = nurbs['coefs'].shape
        
        # 沿 v 方向提升次数
        if ntimes[1] == 0:
            coefs = nurbs['coefs']
            knots2 = nurbs['knots'][1]
        else:
            coefs = nurbs['coefs'].reshape(4*num1, num2)
            coefs, knots2 = bspdegelev(degree[1], coefs, nurbs['knots'][1], ntimes[1])
            num2 = coefs.shape[1]
            coefs = coefs.reshape(4, num1, num2)
        
        # 沿 u 方向提升次数
        if ntimes[0] == 0:
            knots1 = nurbs['knots'][0]
        else:
            coefs = np.transpose(coefs, (0, 2, 1))
            coefs = coefs.reshape(4*num2, num1)
            coefs, knots1 = bspdegelev(degree[0], coefs, nurbs['knots'][0], ntimes[0])
            coefs = coefs.reshape(4, num2, coefs.shape[1])
            coefs = np.transpose(coefs, (0, 2, 1))
        
        knots = [knots1, knots2]
    
    elif len(nurbs['knots']) == 3:  # 体
        dim, num1, num2, num3 = nurbs['coefs'].shape
        
        # 沿 w 方向提升次数
        if ntimes[2] == 0:
            coefs = nurbs['coefs']
            knots3 = nurbs['knots'][2]
        else:
            coefs = nurbs['coefs'].reshape(4*num1*num2, num3)
            coefs, knots3 = bspdegelev(degree[2], coefs, nurbs['knots'][2], ntimes[2])
            num3 = coefs.shape[1]
            coefs = coefs.reshape(4, num1, num2, num3)
        
        # 沿 v 方向提升次数
        if ntimes[1] == 0:
            knots2 = nurbs['knots'][1]
        else:
            coefs = np.transpose(coefs, (0, 1, 3, 2))
            coefs = coefs.reshape(4*num1*num3, num2)
            coefs, knots2 = bspdegelev(degree[1], coefs, nurbs['knots'][1], ntimes[1])
            num2 = coefs.shape[1]
            coefs = coefs.reshape(4, num1, num3, num2)
            coefs = np.transpose(coefs, (0, 1, 3, 2))
        
        # 沿 u 方向提升次数
        if ntimes[0] == 0:
            knots1 = nurbs['knots'][0]
        else:
            coefs = np.transpose(coefs, (0, 2, 3, 1))
            coefs = coefs.reshape(4*num2*num3, num1)
            coefs, knots1 = bspdegelev(degree[0], coefs, nurbs['knots'][0], ntimes[0])
            coefs = coefs.reshape(4, num2, num3, coefs.shape[1])
            coefs = np.transpose(coefs, (0, 3, 1, 2))
        
        knots = [knots1, knots2, knots3]
    
    else:
        raise ValueError('不支持的 NURBS 维度')
    
    # 构造新的 NURBS
    inurbs = nrbmak(coefs, knots)
    
    return inurbs

