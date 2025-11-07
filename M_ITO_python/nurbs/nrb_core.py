"""
NURBS 核心函数
从 MATLAB NURBS Toolbox 转换
"""

import numpy as np
from .bspline import findspan, basisfun, basisfunder, numbasisfun


def bspeval(d, c, k, u):
    """
    在参数点上求值 B-Spline
    
    参数:
        d: B-Spline 的次数
        c: 控制点，大小为 (dim, nc) 的矩阵
        k: 节点序列，大小为 nk 的行向量
        u: 参数求值点，大小为 nu 的行向量
    
    返回:
        p: 求值点，大小为 (dim, nu) 的矩阵
    """
    u = np.atleast_1d(u)
    nu = u.size
    mc, nc = c.shape
    
    s = findspan(nc-1, d, u, k)
    N = basisfun(s, u, d, k)
    
    tmp1 = s - d
    p = np.zeros((mc, nu))
    for i in range(d+1):
        p += N[:, i].reshape(1, -1) * c[:, tmp1+i]
    
    return p


def nrbmak(coefs, knots):
    """
    构造给定控制点和节点的 NURBS 结构
    
    参数:
        coefs: 控制点，可以是笛卡尔坐标或齐次坐标
               对于曲线: (dim, nu)
               对于曲面: (dim, nu, nv)
               对于体: (dim, nu, nv, nw)
        knots: 节点序列
               对于曲线: 向量
               对于曲面/体: 包含 U 和 V (和 W) 的单元数组
    
    返回:
        nurbs: NURBS 数据结构字典
    """
    nurbs = {
        'form': 'B-NURBS',
        'dim': 4,
        'number': [],
        'coefs': None,
        'knots': None,
        'order': []
    }
    
    np_shape = coefs.shape
    dim = np_shape[0]
    
    if isinstance(knots, (list, tuple)):  # 曲面或体
        if len(knots) == 3:  # 体
            if len(np_shape) == 3:
                np_shape = list(np_shape) + [1]
            elif len(np_shape) == 2:
                np_shape = list(np_shape) + [1, 1]
            
            nurbs['number'] = list(np_shape[1:4])
            if dim < 4:
                temp_coefs = np.zeros([4] + list(np_shape[1:4]))
                temp_coefs[0, :, :, :] = 0.0
                temp_coefs[1, :, :, :] = 0.0
                temp_coefs[2, :, :, :] = 0.0
                temp_coefs[3, :, :, :] = 1.0
                temp_coefs[0:dim, :, :, :] = coefs
                nurbs['coefs'] = temp_coefs
            else:
                nurbs['coefs'] = coefs
            
            uorder = len(knots[0]) - np_shape[1]
            vorder = len(knots[1]) - np_shape[2]
            worder = len(knots[2]) - np_shape[3]
            uknots = np.sort(knots[0])
            vknots = np.sort(knots[1])
            wknots = np.sort(knots[2])
            nurbs['knots'] = [uknots, vknots, wknots]
            nurbs['order'] = [uorder, vorder, worder]
            
        elif len(knots) == 2:  # 曲面
            if len(np_shape) == 2:
                np_shape = list(np_shape) + [1]
            
            nurbs['number'] = list(np_shape[1:3])
            if dim < 4:
                temp_coefs = np.zeros([4] + list(np_shape[1:3]))
                temp_coefs[0, :, :] = 0.0
                temp_coefs[1, :, :] = 0.0
                temp_coefs[2, :, :] = 0.0
                temp_coefs[3, :, :] = 1.0
                temp_coefs[0:dim, :, :] = coefs
                nurbs['coefs'] = temp_coefs
            else:
                nurbs['coefs'] = coefs
            
            uorder = len(knots[0]) - np_shape[1]
            vorder = len(knots[1]) - np_shape[2]
            uknots = np.sort(knots[0])
            vknots = np.sort(knots[1])
            nurbs['knots'] = [uknots, vknots]
            nurbs['order'] = [uorder, vorder]
    else:  # 曲线
        nurbs['number'] = [np_shape[1]]
        if dim < 4:
            temp_coefs = np.zeros([4, np_shape[1]])
            temp_coefs[0, :] = 0.0
            temp_coefs[1, :] = 0.0
            temp_coefs[2, :] = 0.0
            temp_coefs[3, :] = 1.0
            temp_coefs[0:dim, :] = coefs
            nurbs['coefs'] = temp_coefs
        else:
            nurbs['coefs'] = coefs
        
        order = len(knots) - np_shape[1]
        nurbs['order'] = order
        knots = np.sort(knots)
        nurbs['knots'] = knots
    
    return nurbs


def nrbeval(nurbs, tt):
    """
    在参数点上求值 NURBS
    
    参数:
        nurbs: NURBS 结构
        tt: 参数求值点
            对于曲线: u 向量
            对于曲面: [u, v] 列表或 (2, n) 数组
            对于体: [u, v, w] 列表或 (3, n) 数组
    
    返回:
        p: 求值点的笛卡尔坐标 (x,y,z)
        w: 齐次坐标的权重
    """
    if not isinstance(nurbs['knots'], list):  # 曲线
        if isinstance(tt, list) and len(tt) == 1:
            tt = tt[0]
        
        tt = np.atleast_1d(tt)
        val = bspeval(nurbs['order']-1, nurbs['coefs'], nurbs['knots'], tt)
        
        w = val[3, :]
        p = val[0:3, :]
        p = p / w.reshape(1, -1)
        
        return p, w
    
    elif len(nurbs['knots']) == 2:  # 曲面
        num1 = nurbs['number'][0]
        num2 = nurbs['number'][1]
        degree = [o-1 for o in nurbs['order']]
        
        if isinstance(tt, list):  # 在 [u,v] 网格上求值
            nt1 = len(tt[0])
            nt2 = len(tt[1])
            
            # 调试：打印输入参数范围
            debug = False  # 设为 True 启用调试
            if debug:
                print(f'[nrbeval grid] tt[0]: [{tt[0][0]:.3f}, {tt[0][-1]:.3f}], len={nt1}')
                print(f'[nrbeval grid] tt[1]: [{tt[1][0]:.3f}, {tt[1][-1]:.3f}], len={nt2}')
                print(f'[nrbeval grid] nurbs.coefs shape: {nurbs["coefs"].shape}')
                print(f'[nrbeval grid] num1={num1}, num2={num2}')
            
            # 沿 v 方向求值（列主序与 MATLAB 一致）
            val = nurbs['coefs'].reshape(4*num1, num2, order='F')
            val = bspeval(degree[1], val, nurbs['knots'][1], tt[1])
            val = val.reshape(4, num1, nt2, order='F')
            
            if debug:
                print(f'[nrbeval grid] After v-eval: val shape={val.shape}')
                print(f'[nrbeval grid] val[0] range: [{np.min(val[0]):.3f}, {np.max(val[0]):.3f}]')
                print(f'[nrbeval grid] val[1] range: [{np.min(val[1]):.3f}, {np.max(val[1]):.3f}]')
            
            # 沿 u 方向求值
            val = np.transpose(val, (0, 2, 1))
            val = val.reshape(4*nt2, num1, order='F')
            val = bspeval(degree[0], val, nurbs['knots'][0], tt[0])
            val = val.reshape(4, nt2, nt1, order='F')
            val = np.transpose(val, (0, 2, 1))
            
            if debug:
                print(f'[nrbeval grid] After u-eval: val shape={val.shape}')
                print(f'[nrbeval grid] val[0] range: [{np.min(val[0]):.3f}, {np.max(val[0]):.3f}]')
                print(f'[nrbeval grid] val[1] range: [{np.min(val[1]):.3f}, {np.max(val[1]):.3f}]')
            
            w = val[3, :, :]
            p = val[0:3, :, :]
            # 除零保护
            w_safe = np.where(np.abs(w) < 1e-10, 1.0, w)
            p = p / w_safe.reshape(1, w_safe.shape[0], w_safe.shape[1])
            
            # Debug output for plot_data debugging
            debug_plot = False  # Set to True to enable debug output
            if debug_plot:
                print(f'[nrbeval grid] After division by w: p[0] range: [{np.min(p[0]):.3f}, {np.max(p[0]):.3f}]')
                print(f'[nrbeval grid] p[1] range: [{np.min(p[1]):.3f}, {np.max(p[1]):.3f}]')
                print(f'[nrbeval grid] p shape: {p.shape}, w shape: {w.shape}')
                print(f'[nrbeval grid] w range: [{np.min(w):.6f}, {np.max(w):.6f}]')
            
            if debug:
                print(f'[nrbeval grid] After division by w: p[0] range: [{np.min(p[0]):.3f}, {np.max(p[0]):.3f}]')
                print(f'[nrbeval grid] p[1] range: [{np.min(p[1]):.3f}, {np.max(p[1]):.3f}]')
            
            return p, w
        else:  # 在散点上求值
            tt = np.atleast_2d(tt)
            st = tt.shape
            if st[0] != 2 and st[1] == 2 and len(st) == 2:
                tt = tt.T
                st = tt.shape
            nt = np.prod(st[1:])
            
            tt = tt.reshape(2, nt)
            
            val = nurbs['coefs'].reshape(4*num1, num2, order='F')
            val = bspeval(degree[1], val, nurbs['knots'][1], tt[1, :])
            val = val.reshape(4, num1, nt, order='F')
            
            # 沿 u 方向求值
            pnts = np.zeros((4, nt))
            for v in range(nt):
                coefs = val[:, :, v].reshape(4, num1, order='F')
                pnts[:, v] = bspeval(degree[0], coefs, nurbs['knots'][0], tt[0, v:v+1]).flatten()
            
            w = pnts[3, :]
            p = pnts[0:3, :]
            # 除零保护
            w_safe = np.where(np.abs(w) < 1e-10, 1.0, w)
            p = p / w_safe.reshape(1, -1)
            
            if len(st) != 2:
                w = w.reshape(st[1:])
                p = p.reshape([3] + list(st[1:]))
            
            return p, w
    
    # 这里可以添加体(volume)的支持，但当前代码主要用于2D曲面
    else:
        raise NotImplementedError("体(volume)求值尚未实现")


def nrbbasisfun(points, nrb):
    """
    NURBS 基函数
    
    参数:
        points: 参数坐标
                对于曲线: u
                对于曲面: [u, v] 或 (2, n) 数组
        nrb: NURBS 结构
    
    返回:
        B: 基函数值，size(B) = [npts, prod(nrb['order'])]
        id: 在每个点非零的基函数索引，size(id) == size(B)
    """
    if not isinstance(nrb['knots'], list):  # NURBS 曲线
        knt = [nrb['knots']]
    else:
        knt = nrb['knots']
    
    ndim = len(nrb['number'])
    w = nrb['coefs'][3, :].reshape(nrb['number'])
    
    sp = []
    N = []
    num = []
    for idim in range(ndim):
        if isinstance(points, list):
            pts_dim = points[idim]
        else:
            pts_dim = points[idim, :]
        
        sp_idim = findspan(nrb['number'][idim]-1, nrb['order'][idim]-1, pts_dim, knt[idim])
        sp.append(sp_idim)
        
        N.append(basisfun(sp[idim], pts_dim, nrb['order'][idim]-1, knt[idim]))
        num.append(numbasisfun(sp[idim], pts_dim, nrb['order'][idim]-1, knt[idim]) + 1)  # +1 转为 1-based
    
    if ndim == 1:
        id_vals = num[0]
        B = w[num[0]-1].reshape(N[0].shape) * N[0]  # -1 转回 0-based
        sum_B = np.sum(B, axis=1, keepdims=True)
        sum_B = np.where(np.abs(sum_B) < 1e-14, 1.0, sum_B)  # 除零保护
        B = B / sum_B
        return B, id_vals
    else:
        if isinstance(points, list):
            npts_dim = [len(p) for p in points]
            cumnpts = np.cumprod([1] + npts_dim)
            npts = np.prod(npts_dim)
            val_aux = 1
            numaux = np.array([[1]])  # 初始化为数组而不是标量
            cumorder = np.cumprod([1] + nrb['order'])
            cumnumber = np.cumprod([1] + nrb['number'])
            
            for idim in range(ndim):
                val_aux = np.kron(N[idim], val_aux)
                num_dim = num[idim].reshape(1, npts_dim[idim], 1, nrb['order'][idim])
                num_dim = np.tile(num_dim, (cumnpts[idim], 1, cumorder[idim], 1))
                
                num_prev = numaux.reshape(cumnpts[idim], 1, cumorder[idim], 1)
                num_prev = np.tile(num_prev, (1, npts_dim[idim], 1, nrb['order'][idim]))
                
                # 转换为线性索引
                # 确保索引在有效范围内
                idx0 = num_prev.flatten().astype(int) - 1  # 转为 0-based
                idx1 = num_dim.flatten().astype(int) - 1    # 转为 0-based
                
                # 边界检查
                idx0 = np.clip(idx0, 0, cumnumber[idim] - 1)
                idx1 = np.clip(idx1, 0, nrb['number'][idim] - 1)
                
                numaux = np.ravel_multi_index(
                    [idx0, idx1],
                    (cumnumber[idim], nrb['number'][idim]),
                    order='F'
                ) + 1  # +1 转回 1-based
                numaux = numaux.reshape(cumnpts[idim+1], cumorder[idim+1])
            
            B = val_aux.reshape(npts, np.prod(nrb['order']))
            id_vals = numaux.reshape(npts, np.prod(nrb['order']))
            # 使用 Fortran 顺序展平以匹配 MATLAB 线性编号
            W = w.flatten(order='F')[id_vals-1]
            WB = W * B
            sum_WB = np.sum(WB, axis=1, keepdims=True)
            sum_WB = np.where(np.abs(sum_WB) < 1e-14, 1.0, sum_WB)  # 除零保护
            B = WB / sum_WB
            
            return B, id_vals
        else:
            npts = points.shape[1]
            B = np.zeros((npts, np.prod(nrb['order'])))
            id_vals = np.zeros((npts, np.prod(nrb['order'])), dtype=int)
            
            for ipt in range(npts):
                val_aux = np.array([1.0])  # 初始化为数组
                local_num = []
                for idim in range(ndim):
                    val_aux = np.outer(val_aux, N[idim][ipt, :]).flatten()
                    local_num.append(num[idim][ipt, :])
                
                # 生成多维网格
                local_num_grid = np.meshgrid(*local_num, indexing='ij')
                # 确保索引在有效范围内
                indices = []
                for idx_dim, ln in enumerate(local_num_grid):
                    idx = ln.flatten().astype(int) - 1  # 转为 0-based
                    idx = np.clip(idx, 0, nrb['number'][idx_dim] - 1)  # 边界检查
                    indices.append(idx)
                
                id_vals[ipt, :] = np.ravel_multi_index(indices, nrb['number'], order='F') + 1  # +1 回 1-based
                
                # 使用 Fortran 顺序展平以匹配 MATLAB 线性编号
                W = w.flatten(order='F')[id_vals[ipt, :]-1].reshape(val_aux.shape)
                val_aux = W * val_aux
                sum_val = np.sum(val_aux)
                sum_val = 1.0 if abs(sum_val) < 1e-14 else sum_val  # 除零保护
                B[ipt, :] = val_aux / sum_val
            
            return B, id_vals


def nrbbasisfunder(points, nrb):
    """
    NURBS 基函数导数
    
    参数:
        points: 参数坐标
        nrb: NURBS 结构
    
    返回:
        对于曲线: Bu, N
        对于曲面: Bu, Bv, N
        对于体: Bu, Bv, Bw, N
    """
    if not isinstance(nrb['knots'], list):  # NURBS 曲线
        knt = [nrb['knots']]
    else:
        knt = nrb['knots']
    
    ndim = len(nrb['number'])
    w = nrb['coefs'][3, :].reshape(nrb['number'])
    
    sp = []
    N = []
    Nder = []
    num = []
    for idim in range(ndim):
        if isinstance(points, list):
            pts_dim = points[idim]
        else:
            pts_dim = points[idim, :]
        
        sp.append(findspan(nrb['number'][idim]-1, nrb['order'][idim]-1, pts_dim, knt[idim]))
        Nprime = basisfunder(sp[idim], nrb['order'][idim]-1, pts_dim, knt[idim], 1)
        N.append(Nprime[:, 0, :])
        Nder.append(Nprime[:, 1, :])
        num.append(numbasisfun(sp[idim], pts_dim, nrb['order'][idim]-1, knt[idim]) + 1)
    
    if ndim == 1:
        B1 = w[num[0]-1].reshape(N[0].shape) * N[0]
        W = np.sum(B1, axis=1, keepdims=True)
        B2 = w[num[0]-1].reshape(N[0].shape) * Nder[0]
        Wder = np.sum(B2, axis=1, keepdims=True)
        
        B2 = B2 / W
        B1 = B1 * (Wder / W**2)
        B = B2 - B1
        return B, num[0]
    else:
        id_vals = nrbnumbasisfun(points, nrb)
        if isinstance(points, list):
            npts_dim = [len(p) for p in points]
            npts = np.prod(npts_dim)
            val_aux = 1
            val_ders = [1 for _ in range(ndim)]
            for idim in range(ndim):
                val_aux = np.kron(N[idim], val_aux)
                for jdim in range(ndim):
                    if idim == jdim:
                        val_ders[idim] = np.kron(Nder[jdim], val_ders[idim])
                    else:
                        val_ders[idim] = np.kron(N[jdim], val_ders[idim])
            
            B1 = w.flatten(order='F')[id_vals-1] * val_aux.reshape(npts, np.prod(nrb['order']))
            W = np.sum(B1, axis=1, keepdims=True)
            results = []
            for idim in range(ndim):
                B2 = w.flatten()[id_vals-1] * val_ders[idim].reshape(npts, np.prod(nrb['order']))
                Wder = np.sum(B2, axis=1, keepdims=True)
                results.append(B2 / W - B1 * (Wder / W**2))
            
            results.append(id_vals)
            return tuple(results)
        else:
            npts = points.shape[1]
            B = np.zeros((npts, np.prod(nrb['order'])))
            Bder = [np.zeros((npts, np.prod(nrb['order']))) for _ in range(ndim)]
            
            for ipt in range(npts):
                val_aux = np.array([1.0])  # 初始化为数组
                val_ders = [np.array([1.0]) for _ in range(ndim)]  # 初始化为数组
                for idim in range(ndim):
                    val_aux = np.outer(val_aux, N[idim][ipt, :]).flatten()
                    for jdim in range(ndim):
                        if idim == jdim:
                            val_ders[idim] = np.outer(val_ders[idim], Nder[jdim][ipt, :]).flatten()
                        else:
                            val_ders[idim] = np.outer(val_ders[idim], N[jdim][ipt, :]).flatten()
                
                wval = w.flatten(order='F')[id_vals[ipt, :]-1].reshape(val_aux.shape)
                val_aux = val_aux * wval
                W = np.sum(val_aux)
                for idim in range(ndim):
                    val_ders[idim] = val_ders[idim] * wval
                    Wder = np.sum(val_ders[idim])
                    Bder[idim][ipt, :] = val_ders[idim] / W - val_aux * (Wder / W**2)
            
            results = Bder + [id_vals]
            return tuple(results)


def nrbnumbasisfun(points, nrb):
    """
    NURBS 基函数编号
    
    参数:
        points: 参数坐标
        nrb: NURBS 结构
    
    返回:
        idx: 在每个点非零的基函数索引
             size(idx) = [npts, prod(nrb['order'])]
    """
    if not isinstance(nrb['knots'], list):  # NURBS 曲线
        iv = findspan(nrb['number'][0]-1, nrb['order']-1, points, nrb['knots'])
        idx = numbasisfun(iv, points, nrb['order']-1, nrb['knots'])
        return idx
    else:
        ndim = len(nrb['number'])
        if isinstance(points, list):
            sp = []
            num = []
            for idim in range(ndim):
                pts_dim = points[idim]
                sp.append(findspan(nrb['number'][idim]-1, nrb['order'][idim]-1, pts_dim, nrb['knots'][idim]))
                num.append(numbasisfun(sp[idim], pts_dim, nrb['order'][idim]-1, nrb['knots'][idim]) + 1)
            
            npts_dim = [len(p) for p in points]
            cumnpts = np.cumprod([1] + npts_dim)
            npts = np.prod(npts_dim)
            
            numaux = np.array([[1]])  # 初始化为数组而不是标量
            cumorder = np.cumprod([1] + nrb['order'])
            cumnumber = np.cumprod([1] + nrb['number'])
            for idim in range(ndim):
                num_dim = num[idim].reshape(1, npts_dim[idim], 1, nrb['order'][idim])
                num_dim = np.tile(num_dim, (cumnpts[idim], 1, cumorder[idim], 1))
                
                num_prev = numaux.reshape(cumnpts[idim], 1, cumorder[idim], 1)
                num_prev = np.tile(num_prev, (1, npts_dim[idim], 1, nrb['order'][idim]))
                
                # 确保索引在有效范围内
                idx0 = num_prev.flatten().astype(int) - 1  # 转为 0-based
                idx1 = num_dim.flatten().astype(int) - 1    # 转为 0-based
                
                # 边界检查
                idx0 = np.clip(idx0, 0, cumnumber[idim] - 1)
                idx1 = np.clip(idx1, 0, nrb['number'][idim] - 1)
                
                numaux = np.ravel_multi_index(
                    [idx0, idx1],
                    (cumnumber[idim], nrb['number'][idim]),
                    order='F'
                ) + 1
                numaux = numaux.reshape(cumnpts[idim+1], cumorder[idim+1])
            
            idx = numaux.reshape(npts, np.prod(nrb['order']))
            return idx
        else:
            sp = []
            num = []
            for idim in range(ndim):
                pts_dim = points[idim, :]
                sp.append(findspan(nrb['number'][idim]-1, nrb['order'][idim]-1, pts_dim, nrb['knots'][idim]))
                num.append(numbasisfun(sp[idim], pts_dim, nrb['order'][idim]-1, nrb['knots'][idim]) + 1)
            
            npts = points.shape[1]
            idx = np.zeros((npts, np.prod(nrb['order'])), dtype=int)
            
            for ipt in range(npts):
                local_num = [num[idim][ipt, :] for idim in range(ndim)]
                local_num_grid = np.meshgrid(*local_num, indexing='ij')
                # 确保索引在有效范围内
                indices = []
                for idx_dim, ln in enumerate(local_num_grid):
                    idx_val = ln.flatten().astype(int) - 1  # 转为 0-based
                    idx_val = np.clip(idx_val, 0, nrb['number'][idx_dim] - 1)  # 边界检查
                    indices.append(idx_val)
                idx[ipt, :] = np.ravel_multi_index(indices, nrb['number'], order='F') + 1
            
            return idx

