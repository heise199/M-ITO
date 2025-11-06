"""
测试基函数在边界点的计算
"""

import numpy as np
from nurbs import nrbbasisfun, nrbmak
from nurbs.bspline import findspan, basisfun

# 创建简单的NURBS曲面
def test_boundary_point():
    # 简单的双线性曲面
    coefs = np.zeros((4, 3, 3))
    # X 坐标
    coefs[0, :, :] = np.array([[0, 5, 10], [0, 5, 10], [0, 5, 10]])
    # Y 坐标
    coefs[1, :, :] = np.array([[0, 0, 0], [2.5, 2.5, 2.5], [5, 5, 5]])
    # Z 坐标
    coefs[2, :, :] = 0
    # 权重
    coefs[3, :, :] = 1
    
    # 节点向量：3次 B-spline with 3 control points -> 需要 3+3+1=7 个节点
    knots_u = np.array([0, 0, 0, 1, 1, 1])
    knots_v = np.array([0, 0, 0, 1, 1, 1])
    
    NURBS = nrbmak(coefs, [knots_u, knots_v])
    
    print(f'NURBS.number: {NURBS["number"]}')
    print(f'NURBS.order: {NURBS["order"]}')
    print(f'NURBS.knots[0]: {NURBS["knots"][0]}')
    print(f'NURBS.knots[1]: {NURBS["knots"][1]}')
    
    # 测试不同的参数点
    test_points = [
        (0.0, 0.5, "左边界-中间"),
        (0.5, 0.5, "中心"),
        (1.0, 0.5, "右边界-中间"),
        (0.99, 0.5, "接近右边界"),
    ]
    
    for u, v, desc in test_points:
        print(f'\n测试点: u={u}, v={v} ({desc})')
        
        # 测试 findspan
        try:
            n_u = NURBS['number'][0] - 1  # n = 控制点数 - 1
            p_u = NURBS['order'][0] - 1   # p = 次数
            span_u = findspan(n_u, p_u, u, NURBS['knots'][0])
            print(f'  findspan(u): n={n_u}, p={p_u}, span={span_u}')
            print(f'  节点区间: [{NURBS["knots"][0][span_u]}, {NURBS["knots"][0][span_u+1]}]')
            
            # 测试 basisfun
            N_u = basisfun(span_u, u, p_u, NURBS['knots'][0])
            print(f'  basisfun(u): {N_u}')
            print(f'  sum(N_u) = {np.sum(N_u)}')
            
        except Exception as e:
            print(f'  错误: {e}')
            import traceback
            traceback.print_exc()
        
        # 测试完整的 nrbbasisfun
        try:
            N, id_vals = nrbbasisfun(np.array([[u], [v]]), NURBS)
            print(f'  nrbbasisfun: N.shape={N.shape}, id.shape={id_vals.shape}')
            print(f'  N = {N.flatten()}')
            print(f'  id = {id_vals.flatten()}')
            print(f'  sum(N) = {np.sum(N)}')
        except Exception as e:
            print(f'  nrbbasisfun 错误: {e}')
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    test_boundary_point()

