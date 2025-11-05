"""
NURBS基函数导数计算模块 - 完全1:1匹配MATLAB的nrbbasisfunder实现
"""
import numpy as np
from utils import find_span, basis_function_derivatives


def nrbnumbasisfun(points, nrb):
    """
    计算NURBS基函数的编号（控制点索引）
    完全匹配MATLAB的nrbnumbasisfun实现
    
    参数:
        points: 参数坐标点集，shape (2, n_points)
        nrb: NURBS曲面对象
    
    返回:
        idx: 控制点索引，shape (n_points, prod(nrb.order))
    """
    from utils import numbasisfun
    
    npts = points.shape[1]
    idx = np.zeros((npts, (nrb.degree_u + 1) * (nrb.degree_v + 1)), dtype=int)
    
    for ipt in range(npts):
        u_val = points[0, ipt]
        v_val = points[1, ipt]
        
        # 计算span
        sp_u = find_span(nrb.ctrlpts_size_u - 1, nrb.degree_u, u_val, np.array(nrb.knotvector_u))
        sp_v = find_span(nrb.ctrlpts_size_v - 1, nrb.degree_v, v_val, np.array(nrb.knotvector_v))
        
        # 计算局部索引
        # MATLAB: num{idim} = numbasisfun(sp{idim}, pts_dim, nrb.order(idim)-1, knt{idim}) + 1;
        # numbasisfun返回的是控制点索引（相对于span的局部索引），然后+1得到MATLAB索引
        num_u_raw = numbasisfun([sp_u], [u_val], nrb.degree_u, np.array(nrb.knotvector_u))[0]
        num_v_raw = numbasisfun([sp_v], [v_val], nrb.degree_v, np.array(nrb.knotvector_v))[0]
        num_u = num_u_raw + 1  # 转换为MATLAB索引（从1开始）
        num_v = num_v_raw + 1  # 转换为MATLAB索引（从1开始）
        
        # 调试第一个点
        if ipt == 0:
            print(f"\n=== nrbnumbasisfun Debug (point {ipt}) ===")
            print(f"u_val={u_val}, v_val={v_val}")
            print(f"sp_u={sp_u}, sp_v={sp_v}")
            print(f"num_u_raw={num_u_raw}, num_v_raw={num_v_raw}")
            print(f"num_u={num_u}, num_v={num_v}")
            print(f"ctrlpts_size_u={nrb.ctrlpts_size_u}, ctrlpts_size_v={nrb.ctrlpts_size_v}")
        
        # 计算全局索引（tensor product）
        # MATLAB: idx(ipt,:) = reshape(sub2ind(nrb.number, local_num{:}), 1, size(idx, 2))
        # MATLAB的sub2ind([num_u, num_v], u, v) = (v-1)*num_u + u（列优先！）
        # 注意：num_u和num_v已经是MATLAB索引（从1开始）
        idx_count = 0
        for v_idx_local in range(nrb.degree_v + 1):
            for u_idx_local in range(nrb.degree_u + 1):
                u_global = num_u[u_idx_local]  # MATLAB索引（从1开始）
                v_global = num_v[v_idx_local]  # MATLAB索引（从1开始）
                # 关键修复：MATLAB的sub2ind是列优先：(v-1)*num_u + u
                idx[ipt, idx_count] = (v_global - 1) * nrb.ctrlpts_size_u + u_global
                if ipt == 0:
                    print(f"  u_idx_local={u_idx_local}, v_idx_local={v_idx_local}")
                    print(f"  u_global={u_global}, v_global={v_global}")
                    print(f"  idx[{ipt}, {idx_count}] = ({v_global}-1)*{nrb.ctrlpts_size_u}+{u_global} = {idx[ipt, idx_count]}")
                idx_count += 1
    
    return idx


def nrbbasisfunder(uv, NURBS):
    """
    计算NURBS基函数对参数u和v的导数
    完全1:1匹配MATLAB的nrbbasisfunder实现
    
    参数:
        uv: 参数坐标点集，shape (2, n_points)
        NURBS: NURBS曲面对象
    
    返回:
        dRu: 基函数对u的导数，shape (n_points, n_basis)
        dRv: 基函数对v的导数，shape (n_points, n_basis)
        id_array: 控制点索引，shape (n_points, n_basis)，MATLAB索引（从1开始）
    
    完全复刻MATLAB的nrbbasisfunder实现（nurbs-1.3.13/inst/nrbbasisfunder.m）
    """
    npts = uv.shape[1]
    ndim = 2  # 曲面是2维
    
    # 获取权重：MATLAB: w = reshape(nrb.coefs(4,:), [nrb.number 1])
    # NURBS.coefs shape是(4, num_u, num_v)
    # MATLAB的reshape是按列填充的，所以需要按列优先展平
    num_u = NURBS.ctrlpts_size_u
    num_v = NURBS.ctrlpts_size_v
    # 关键修复：与pre_iga.py中的坐标提取方式保持一致
    # MATLAB: sub2ind([num_u, num_v], u, v) = u + (v-1)*num_u（列优先！）
    # coefs(4, :) 按列展平：coefs(4,1,1), coefs(4,2,1), ..., coefs(4,num_u,1), coefs(4,1,2), ...
    w = np.zeros(num_u * num_v)
    for v_idx in range(num_v):
        for u_idx in range(num_u):
            # MATLAB: idx = u + (v-1)*num_u，Python（0-based）: idx = u_idx + v_idx*num_u
            idx = u_idx + v_idx * num_u
            w[idx] = NURBS.coefs[3, u_idx, v_idx]
    
    # 计算每个方向的span和基函数
    knotvector_u = np.array(NURBS.knotvector_u)
    knotvector_v = np.array(NURBS.knotvector_v)
    
    # 存储每个方向的基函数值和导数
    N = {}  # 基函数值
    Nder = {}  # 基函数导数
    sp = {}  # span索引
    
    # u方向
    pts_u = uv[0, :]
    sp[0] = []
    for u_val in pts_u:
        sp[0].append(find_span(num_u - 1, NURBS.degree_u, u_val, knotvector_u))
    
    # 计算u方向的基函数和导数
    Nprime_u = []
    for ipt in range(npts):
        u_val = pts_u[ipt]
        span_u = sp[0][ipt]
        # basisfunder返回shape (nders+1, p+1)
        ders = basis_function_derivatives(NURBS.degree_u, knotvector_u, span_u, u_val, nders=1)
        Nprime_u.append(ders)
        # 调试第一个点
        if ipt == 0:
            print(f"  u_val={u_val}, span_u={span_u}, degree_u={NURBS.degree_u}")
            print(f"  knotvector_u[{span_u-NURBS.degree_u}:{span_u+NURBS.degree_u+2}] = {knotvector_u[max(0,span_u-NURBS.degree_u):min(len(knotvector_u), span_u+NURBS.degree_u+2)]}")
            print(f"  ders shape={ders.shape}, ders[0, :]={ders[0, :]}, ders[1, :]={ders[1, :]}")
    
    # 重塑为 (npts, degree_u+1)
    N[0] = np.zeros((npts, NURBS.degree_u + 1))
    Nder[0] = np.zeros((npts, NURBS.degree_u + 1))
    for ipt in range(npts):
        N[0][ipt, :] = Nprime_u[ipt][0, :]  # 0阶导数（基函数值）
        Nder[0][ipt, :] = Nprime_u[ipt][1, :]  # 1阶导数
    
    # v方向
    pts_v = uv[1, :]
    sp[1] = []
    for v_val in pts_v:
        sp[1].append(find_span(num_v - 1, NURBS.degree_v, v_val, knotvector_v))
    
    # 计算v方向的基函数和导数
    Nprime_v = []
    for ipt in range(npts):
        v_val = pts_v[ipt]
        span_v = sp[1][ipt]
        # basisfunder返回shape (nders+1, p+1)
        ders = basis_function_derivatives(NURBS.degree_v, knotvector_v, span_v, v_val, nders=1)
        Nprime_v.append(ders)
    
    # 重塑为 (npts, degree_v+1)
    N[1] = np.zeros((npts, NURBS.degree_v + 1))
    Nder[1] = np.zeros((npts, NURBS.degree_v + 1))
    for ipt in range(npts):
        N[1][ipt, :] = Nprime_v[ipt][0, :]  # 0阶导数（基函数值）
        Nder[1][ipt, :] = Nprime_v[ipt][1, :]  # 1阶导数
    
    # 获取控制点索引：MATLAB: id = nrbnumbasisfun(points, nrb)
    id = nrbnumbasisfun(uv, NURBS)
    
    # 调试：检查第一个点的基函数值
    if npts > 0:
        debug_ipt = 0
        print(f"\n=== nrbbasisfunder Debug (point {debug_ipt}) ===")
        print(f"u={pts_u[debug_ipt]}, v={pts_v[debug_ipt]}")
        print(f"span_u={sp[0][debug_ipt]}, span_v={sp[1][debug_ipt]}")
        print(f"N[0][{debug_ipt}, :] = {N[0][debug_ipt, :]}")
        print(f"N[1][{debug_ipt}, :] = {N[1][debug_ipt, :]}")
        print(f"Nder[0][{debug_ipt}, :] = {Nder[0][debug_ipt, :]}")
        print(f"Nder[1][{debug_ipt}, :] = {Nder[1][debug_ipt, :]}")
        print(f"id[{debug_ipt}, :] = {id[debug_ipt, :]}")
        print(f"w[id[{debug_ipt}, :]-1] = {w[id[debug_ipt, :] - 1]}")
        print(f"w shape={w.shape}, w range=[{w.min()}, {w.max()}]")
    
    # 计算基函数导数（按MATLAB的逻辑）
    # MATLAB代码（第127-154行）：
    #   for ipt = 1:npts
    #     val_aux = 1;
    #     val_ders = repmat({1}, ndim, 1);
    #     for idim = 1:ndim
    #       val_aux = reshape(val_aux.' * N{idim}(ipt,:), 1, []);
    #       for jdim = 1:ndim
    #         if (idim == jdim)
    #           val_ders{idim} = reshape(val_ders{idim}.' * Nder{jdim}(ipt,:), 1, []);
    #         else
    #           val_ders{idim} = reshape(val_ders{idim}.' * N{jdim}(ipt,:), 1, []);
    #         end
    #       end
    #     end
    #     wval = reshape(w(id(ipt,:)), size(val_aux));
    #     val_aux = val_aux .* wval;
    #     W = sum(val_aux);
    #     for idim = 1:ndim
    #       val_ders{idim} = val_ders{idim} .* wval;
    #       Wder = sum(val_ders{idim});
    #       Bder{idim}(ipt,:) = bsxfun(@(x,y) x./y, val_ders{idim}, W) - bsxfun(@(x,y) x.*y, val_aux, Wder ./ W.^2);
    #     end
    #   end
    
    n_basis = (NURBS.degree_u + 1) * (NURBS.degree_v + 1)
    dRu = np.zeros((npts, n_basis))
    dRv = np.zeros((npts, n_basis))
    
    for ipt in range(npts):
        # MATLAB代码：
        # val_aux = 1;
        # for idim = 1:ndim
        #   val_aux = reshape(val_aux.' * N{idim}(ipt,:), 1, []);
        # end
        # 对于2D：
        # idim=1 (u方向): val_aux = reshape([1].' * N{0}(ipt,:), 1, []) = N{0}(ipt,:)
        # idim=2 (v方向): val_aux = reshape(N{0}(ipt,:).' * N{1}(ipt,:), 1, []) = kron(N{0}, N{1})
        # MATLAB: val_aux = 1; for idim = 1:ndim, val_aux = reshape(val_aux.' * N{idim}(ipt,:), 1, []); end
        # val_aux.' * N{idim} 是外积（列向量 * 行向量）
        # reshape(..., 1, []) 是按行展平（C风格）
        val_aux = np.ones(1)
        for idim in range(ndim):
            # val_aux.' * N{idim}(ipt,:) 相当于 np.outer(val_aux, N[idim][ipt, :])
            # reshape按行展平：flatten('C')
            outer_result = np.outer(val_aux, N[idim][ipt, :])
            val_aux = outer_result.flatten('C')
            if ipt == 0 and idim == 0:
                print(f"  After idim={idim}: val_aux shape={val_aux.shape}, val_aux={val_aux}")
            if ipt == 0 and idim == 1:
                print(f"  After idim={idim}: val_aux shape={val_aux.shape}, val_aux={val_aux}")
        
        # 计算导数
        # MATLAB代码（关键部分）：
        # val_ders = repmat({1}, ndim, 1);
        # for idim = 1:ndim
        #   val_aux = reshape(val_aux.' * N{idim}(ipt,:), 1, []);
        #   for jdim = 1:ndim
        #     if (idim == jdim)
        #       val_ders{idim} = reshape(val_ders{idim}.' * Nder{jdim}(ipt,:), 1, []);
        #     else
        #       val_ders{idim} = reshape(val_ders{idim}.' * N{jdim}(ipt,:), 1, []);
        #     end
        #   end
        # end
        # 注意：val_ders的更新是在idim循环内部，对于每个idim都要更新所有val_ders{idim}
        # 对于u方向（idim=1）：idim=1时用Nder{0}，idim=2时用N{1} -> kron(Nder{0}, N{1})
        # 对于v方向（idim=2）：idim=1时用N{0}，idim=2时用Nder{1} -> kron(N{0}, Nder{1})
        val_ders = [np.ones(1), np.ones(1)]  # [u方向, v方向]
        
        # 按照MATLAB的逻辑：外层循环是idim，内层循环是jdim
        # 注意：val_ders{idim}中的idim是指导数方向，不是循环变量idim
        # 对于u方向导数（varargout{1}）：idim=1时用Nder{1}，idim=2时用N{2}
        # 对于v方向导数（varargout{2}）：idim=1时用N{1}，idim=2时用Nder{2}
        for idim in range(ndim):
            # 更新val_aux（已经在上面完成了）
            # 更新val_ders
            for der_dim in range(ndim):  # der_dim是导数方向索引（0=u方向，1=v方向）
                if idim == der_dim:
                    # 使用导数
                    val_ders[der_dim] = np.outer(val_ders[der_dim], Nder[idim][ipt, :]).flatten('C')
                else:
                    # 使用基函数值
                    val_ders[der_dim] = np.outer(val_ders[der_dim], N[idim][ipt, :]).flatten('C')
        
        val_ders_u = val_ders[0]
        val_ders_v = val_ders[1]
        
        # 获取权重
        # MATLAB: wval = reshape(w(id(ipt,:)), size(val_aux));
        # id(ipt,:)是控制点索引（MATLAB索引，从1开始）
        # MATLAB: w = reshape(nrb.coefs(4,:), [nrb.number 1])
        # coefs(4,:)是按列展平的，reshape后w的形状是[num_u, num_v]
        # 在Python中，coefs[3, :, :]的shape是(num_u, num_v)，flatten('F')后按列展平
        # id(ipt,:)是控制点索引，范围是1到num_u*num_v
        # 需要转换为Python索引（从0开始）
        ctrlpt_indices = id[ipt, :] - 1  # MATLAB索引从1开始，Python从0开始
        wval = w[ctrlpt_indices]  # 从权重数组中获取对应的权重值
        if ipt == 0:
            print(f"  ctrlpt_indices = {ctrlpt_indices}")
            print(f"  w[ctrlpt_indices] = {wval}")
            print(f"  w shape = {w.shape}, id[{ipt}, :] = {id[ipt, :]}")
            # 验证权重提取：检查前几个控制点的权重
            print(f"  w[0:10] = {w[:min(10, len(w))]}")
            print(f"  w非零值数量: {np.count_nonzero(w)} / {len(w)}")
        
        # MATLAB: val_aux = val_aux .* wval;
        val_aux_weighted = val_aux * wval
        W = np.sum(val_aux_weighted)
        if ipt == 0:
            print(f"  val_aux (before weight) = {val_aux}")
            print(f"  wval = {wval}")
            print(f"  val_aux_weighted = {val_aux_weighted}")
            print(f"  W = {W}")
        
        # 计算u方向导数
        # MATLAB: val_ders{idim} = val_ders{idim} .* wval;
        val_ders_u_weighted = val_ders_u * wval
        Wder_u = np.sum(val_ders_u_weighted)
        
        # MATLAB: Bder{idim}(ipt,:) = bsxfun(@(x,y) x./y, val_ders{idim}, W) - bsxfun(@(x,y) x.*y, val_aux, Wder ./ W.^2);
        # 即：val_ders/W - val_aux * (Wder/W^2)
        if W > 1e-15:
            dRu[ipt, :] = val_ders_u_weighted / W - val_aux_weighted * (Wder_u / (W ** 2))
        else:
            # 如果权重和为0，设置导数为0
            dRu[ipt, :] = 0.0
        
        # 计算v方向导数
        val_ders_v_weighted = val_ders_v * wval
        Wder_v = np.sum(val_ders_v_weighted)
        if W > 1e-15:
            dRv[ipt, :] = val_ders_v_weighted / W - val_aux_weighted * (Wder_v / (W ** 2))
        else:
            # 如果权重和为0，设置导数为0
            dRv[ipt, :] = 0.0
    
    return dRu, dRv, id
