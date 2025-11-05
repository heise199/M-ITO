"""
NURBS基函数计算模块 - 计算基函数值和索引
"""
import numpy as np
from utils import find_span, basis_function


def nrbbasisfun(uv, NURBS):
    """
    计算NURBS基函数值和对应的控制点索引
    
    参数:
        uv: 参数坐标，可以是单个点或点集
            单个点: shape (2, 1) 或 (2,)
            点集: shape (2, n) 或字典 {u坐标数组, v坐标数组}
        NURBS: NURBS曲面对象
    
    返回:
        N: 基函数值矩阵，shape (n_points, n_basis)
        id: 对应的控制点索引，shape (n_points, n_basis)
    """
    if isinstance(uv, dict):
        # 点集情况 - 字典格式 {0: u_array, 1: v_array}
        u_array = uv[0] if 0 in uv else uv.get('u', [])
        v_array = uv[1] if 1 in uv else uv.get('v', [])
        
        n_points = len(u_array) * len(v_array)
        n_basis = (NURBS.degree_u + 1) * (NURBS.degree_v + 1)
        
        N = np.zeros((n_points, n_basis))
        id_array = np.zeros((n_points, n_basis), dtype=int)
        
        idx = 0
        # MATLAB的tensor product顺序：kron(N{v}, N{u})意味着u方向变化最快
        # 所以遍历顺序：外层v，内层u
        for v_val in v_array:
            for u_val in u_array:
                # 找到节点区间
                span_u = find_span(NURBS.ctrlpts_size_u - 1, 
                                  NURBS.degree_u, u_val, np.array(NURBS.knotvector_u))
                span_v = find_span(NURBS.ctrlpts_size_v - 1, 
                                  NURBS.degree_v, v_val, np.array(NURBS.knotvector_v))
                
                # 计算基函数
                basis_u = basis_function(NURBS.degree_u, np.array(NURBS.knotvector_u), span_u, u_val)
                basis_v = basis_function(NURBS.degree_v, np.array(NURBS.knotvector_v), span_v, v_val)
                
                # 计算NURBS基函数（考虑权重）
                ctrlpt_idx = 0
                for v_idx in range(span_v - NURBS.degree_v, span_v + 1):
                    for u_idx in range(span_u - NURBS.degree_u, span_u + 1):
                        # 获取控制点权重
                        ctrlpt_4d = NURBS.ctrlpts[v_idx * NURBS.ctrlpts_size_u + u_idx]
                        w = ctrlpt_4d[3]
                        
                        # NURBS基函数
                        N[idx, ctrlpt_idx] = basis_u[u_idx - (span_u - NURBS.degree_u)] * \
                                            basis_v[v_idx - (span_v - NURBS.degree_v)] * w
                        
                        # 控制点索引（MATLAB索引从1开始）
                        # MATLAB的sub2ind([num_u, num_v], u, v) = (v-1)*num_u + u（列优先！）
                        u_idx_matlab = u_idx + 1
                        v_idx_matlab = v_idx + 1
                        id_array[idx, ctrlpt_idx] = (v_idx_matlab - 1) * NURBS.ctrlpts_size_u + u_idx_matlab
                        ctrlpt_idx += 1
                
                # 归一化（除以所有权重的加权和）
                W_sum = np.sum(N[idx, :])
                if W_sum > 0:
                    N[idx, :] = N[idx, :] / W_sum
                
                idx += 1
        
        return N, id_array
    
    else:
        # 单个点情况
        if uv.shape[0] == 2 and uv.shape[1] == 1:
            u_val = uv[0, 0]
            v_val = uv[1, 0]
        elif uv.shape[0] == 2:
            u_val = uv[0]
            v_val = uv[1]
        else:
            raise ValueError("输入格式不正确")
        
        # 找到节点区间
        n_u = NURBS.ctrlpts_size_u - 1
        n_v = NURBS.ctrlpts_size_v - 1
        span_u = find_span(n_u, NURBS.degree_u, u_val, np.array(NURBS.knotvector_u))
        span_v = find_span(n_v, NURBS.degree_v, v_val, np.array(NURBS.knotvector_v))
        
        # 计算基函数
        basis_u = basis_function(NURBS.degree_u, np.array(NURBS.knotvector_u), span_u, u_val)
        basis_v = basis_function(NURBS.degree_v, np.array(NURBS.knotvector_v), span_v, v_val)
        
        n_basis = (NURBS.degree_u + 1) * (NURBS.degree_v + 1)
        N = np.zeros(n_basis)
        id_array = np.zeros(n_basis, dtype=int)
        
        ctrlpt_idx = 0
        # MATLAB中的顺序：先v方向，再u方向
        # for v_idx in range(span_v - degree_v, span_v + 1):
        #     for u_idx in range(span_u - degree_u, span_u + 1):
        # 这与nrbbasisfunder中的顺序必须完全一致
        
        for v_idx in range(span_v - NURBS.degree_v, span_v + 1):
            for u_idx in range(span_u - NURBS.degree_u, span_u + 1):
                # 检查索引是否在有效范围内
                if v_idx < 0 or v_idx >= NURBS.ctrlpts_size_v or \
                   u_idx < 0 or u_idx >= NURBS.ctrlpts_size_u:
                    continue
                
                # 计算基函数数组中的索引
                basis_u_idx = u_idx - (span_u - NURBS.degree_u)
                basis_v_idx = v_idx - (span_v - NURBS.degree_v)
                
                # 检查索引是否在基函数数组范围内
                if basis_u_idx < 0 or basis_u_idx >= len(basis_u) or \
                   basis_v_idx < 0 or basis_v_idx >= len(basis_v):
                    continue
                
                # 获取控制点权重
                ctrlpt_4d = NURBS.ctrlpts[v_idx * NURBS.ctrlpts_size_u + u_idx]
                w = ctrlpt_4d[3]
                
                # NURBS基函数
                N[ctrlpt_idx] = basis_u[basis_u_idx] * basis_v[basis_v_idx] * w
                
                # 控制点索引（MATLAB索引从1开始）
                # MATLAB的sub2ind([num_u, num_v], u, v) = u + (v-1)*num_u（列优先）
                u_idx_matlab = u_idx + 1
                v_idx_matlab = v_idx + 1
                id_array[ctrlpt_idx] = (v_idx_matlab - 1) * NURBS.ctrlpts_size_u + u_idx_matlab
                ctrlpt_idx += 1
        
        # 如果有效的控制点数量少于预期，调整数组大小
        if ctrlpt_idx < n_basis:
            N = N[:ctrlpt_idx]
            id_array = id_array[:ctrlpt_idx]
        
        # 归一化
        W_sum = np.sum(N)
        if W_sum > 0:
            N = N / W_sum
        
        return N.reshape(1, -1), id_array.reshape(1, -1)

