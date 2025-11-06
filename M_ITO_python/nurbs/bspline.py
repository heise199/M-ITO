"""
B-Spline 基础函数
从 MATLAB NURBS Toolbox 转换
"""

import numpy as np


def findspan(n, p, u, U):
    """
    找到 B-Spline 节点向量中参数点的跨度
    
    参数:
        n: 控制点数量 - 1
        p: 样条次数
        u: 参数点 (标量或数组)
        U: 节点序列
    
    返回:
        s: 节点跨度索引 (与 u 相同形状)
    
    改编自 'The NURBS BOOK' pg68 Algorithm A2.1 和 MATLAB NURBS Toolbox
    """
    # 记住输入是否为标量
    is_scalar = np.isscalar(u)
    u = np.atleast_1d(u)
    U = np.asarray(U)
    
    # 容差检查（添加小容差）
    tol = 1e-10
    if np.max(u) > U[-1] + tol or np.min(u) < U[0] - tol:
        raise ValueError(f'某些参数值超出了节点跨度范围: u in [{np.min(u)}, {np.max(u)}], U in [{U[0]}, {U[-1]}]')
    
    # 将超出范围的值裁剪到有效范围内
    u = np.clip(u, U[0], U[-1])
    
    s = np.zeros(u.shape, dtype=int)
    
    for j in range(u.size):
        uu = u.flat[j]
        
        # 特殊情况：参数值在右边界
        if uu >= U[n+1]:
            s.flat[j] = n
            continue
        
        # 线性搜索（更可靠，虽然慢一点）
        # 找到满足 U[i] <= uu < U[i+1] 的 i
        s.flat[j] = p  # 默认值
        for i in range(p, n+1):
            if U[i] <= uu < U[i+1]:
                s.flat[j] = i
                break
            elif i == n and uu >= U[i]:
                # 右边界情况
                s.flat[j] = n
                break
    
    # 如果输入是标量，返回标量；否则返回数组
    return s.item() if is_scalar else s


def basisfun(iv, uv, p, U):
    """
    B-Spline 基函数
    
    参数:
        iv: 节点跨度 (从 findspan() 获得)
        uv: 参数点
        p: 样条次数
        U: 节点序列
    
    返回:
        B: 基函数矩阵 (len(uv) x (p+1))
    
    改编自 'The NURBS BOOK' pg70 Algorithm A2.2
    """
    uv = np.atleast_1d(uv)
    iv = np.atleast_1d(iv)
    
    B = np.zeros((uv.size, p+1))
    
    for jj in range(uv.size):
        i = iv[jj]  # findspan 使用 0-based 编号
        u = uv[jj]
        
        left = np.zeros(p+1)
        right = np.zeros(p+1)
        N = np.zeros(p+1)
        
        N[0] = 1.0
        for j in range(1, p+1):
            left[j] = u - U[i+1-j]
            right[j] = U[i+j] - u
            saved = 0.0
            
            for r in range(j):
                denom = right[r+1] + left[j-r]
                if abs(denom) < 1e-14:  # Division by zero protection
                    # Special case: at knot boundaries with multiplicity
                    # Use 0/0 = 0 convention (standard in NURBS)
                    if abs(right[r+1]) < 1e-14 and abs(left[j-r]) < 1e-14:
                        # Both numerator terms will be zero, keep N[r]
                        temp = 0.0
                    else:
                        temp = 0.0
                else:
                    temp = N[r] / denom
                N[r] = saved + right[r+1] * temp
                saved = left[j-r] * temp
            
            N[j] = saved
        
        B[jj, :] = N
    
    return B


def basisfunder(ii, pl, uu, u_knotl, nders):
    """
    B-Spline 基函数导数
    
    参数:
        ii: 节点跨度索引 (参见 findspan)
        pl: 曲线次数
        uu: 参数点
        u_knotl: 节点向量
        nders: 要计算的导数数量
    
    返回:
        dersv: dersv[n, i, :] 是第 n 个点处的第 (i-1) 阶导数
    
    改编自 'The NURBS BOOK' pg72 Algorithm A2.3
    """
    uu = np.atleast_1d(uu)
    ii = np.atleast_1d(ii)
    
    dersv = np.zeros((uu.size, nders+1, pl+1))
    
    for jj in range(uu.size):
        i = ii[jj]  # 转换为 base-0 编号
        u = uu[jj]
        
        ders = np.zeros((nders+1, pl+1))
        ndu = np.zeros((pl+1, pl+1))
        left = np.zeros(pl+1)
        right = np.zeros(pl+1)
        a = np.zeros((2, pl+1))
        
        ndu[0, 0] = 1.0
        for j in range(1, pl+1):
            left[j] = u - u_knotl[i+1-j]
            right[j] = u_knotl[i+j] - u
            saved = 0.0
            for r in range(j):
                ndu[j, r] = right[r+1] + left[j-r]
                if abs(ndu[j, r]) < 1e-14:  # 除零保护
                    temp = 0.0
                else:
                    temp = ndu[r, j-1] / ndu[j, r]
                ndu[r, j] = saved + right[r+1] * temp
                saved = left[j-r] * temp
            ndu[j, j] = saved
        
        for j in range(pl+1):
            ders[0, j] = ndu[j, pl]
        
        for r in range(pl+1):
            s1 = 0
            s2 = 1
            a[0, 0] = 1.0
            for k in range(1, nders+1):
                d = 0.0
                rk = r - k
                pk = pl - k
                if r >= k:
                    if abs(ndu[pk+1, rk]) < 1e-14:  # 除零保护
                        a[s2, 0] = 0.0
                    else:
                        a[s2, 0] = a[s1, 0] / ndu[pk+1, rk]
                    d = a[s2, 0] * ndu[rk, pk]
                
                if rk >= -1:
                    j1 = 1
                else:
                    j1 = -rk
                
                if r-1 <= pk:
                    j2 = k-1
                else:
                    j2 = pl-r
                
                for j in range(j1, j2+1):
                    if abs(ndu[pk+1, rk+j]) < 1e-14:  # 除零保护
                        a[s2, j] = 0.0
                    else:
                        a[s2, j] = (a[s1, j] - a[s1, j-1]) / ndu[pk+1, rk+j]
                    d += a[s2, j] * ndu[rk+j, pk]
                
                if r <= pk:
                    if abs(ndu[pk+1, r]) < 1e-14:  # 除零保护
                        a[s2, k] = 0.0
                    else:
                        a[s2, k] = -a[s1, k-1] / ndu[pk+1, r]
                    d += a[s2, k] * ndu[r, pk]
                
                ders[k, r] = d
                j = s1
                s1 = s2
                s2 = j
        
        r = pl
        for k in range(1, nders+1):
            for j in range(pl+1):
                ders[k, j] *= r
            r *= (pl - k)
        
        dersv[jj, :, :] = ders
    
    return dersv


def numbasisfun(iv, uv, p, U):
    """
    列出给定节点跨度内的非零 B-Spline 基函数编号
    
    参数:
        iv: 节点跨度 (从 findspan() 获得)
        uv: 参数点
        p: 样条次数
        U: 节点序列
    
    返回:
        B: 基函数编号 (len(uv) x (p+1))
    """
    iv = np.atleast_1d(iv)
    uv = np.atleast_1d(uv)
    
    # 创建 (p+1) 列的索引矩阵
    B = iv[:, np.newaxis] - p + np.arange(p+1)
    
    return B

