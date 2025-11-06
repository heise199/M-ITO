"""
测试实际NURBS的基函数
"""

import numpy as np
from geom_mod import geom_mod
from pre_iga import pre_iga
from nurbs import nrbbasisfun

def test_real_nurbs():
    # 使用与实际问题相同的参数
    L = 10
    W = 5
    Order = [1, 1]
    Num = [11, 6]
    BoundCon = 1
    
    print('生成几何模型...')
    NURBS = geom_mod(L, W, Order, Num, BoundCon)
    
    print(f'\nNURBS 结构:')
    print(f'  number: {NURBS["number"]}')
    print(f'  order: {NURBS["order"]}')
    print(f'  knots[0]长度: {len(NURBS["knots"][0])}, 范围: [{NURBS["knots"][0][0]}, {NURBS["knots"][0][-1]}]')
    print(f'  knots[1]长度: {len(NURBS["knots"][1])}, 范围: [{NURBS["knots"][1][0]}, {NURBS["knots"][1][-1]}]')
    print(f'  knots[0]: {NURBS["knots"][0][:10]} ... {NURBS["knots"][0][-5:]}')
    print(f'  knots[1]: {NURBS["knots"][1][:10]} ... {NURBS["knots"][1][-5:]}')
    print(f'  coefs shape: {NURBS["coefs"].shape}')
    
    # 检查权重
    weights = NURBS['coefs'][3, :, :].flatten(order='F')
    print(f'\n权重检查:')
    print(f'  权重范围: [{np.min(weights)}, {np.max(weights)}]')
    print(f'  权重平均: {np.mean(weights)}')
    print(f'  零权重数量: {np.sum(np.abs(weights) < 1e-10)}')
    
    # 测试载荷点
    load_u = 1.0
    load_v = 0.5
    
    print(f'\n测试载荷点: u={load_u}, v={load_v}')
    
    try:
        N, id_vals = nrbbasisfun(np.array([[load_u], [load_v]]), NURBS)
        print(f'  N.shape: {N.shape}')
        print(f'  N 非零数量: {np.count_nonzero(N)}')
        print(f'  N: {N.flatten()}')
        print(f'  sum(N): {np.sum(N)}')
        print(f'  id_vals: {id_vals.flatten()}')
        
        # 检查对应的权重
        ids = id_vals.flatten() - 1  # 转为 0-based
        corresponding_weights = weights[ids]
        print(f'  对应控制点的权重: {corresponding_weights}')
        
    except Exception as e:
        print(f'  错误: {e}')
        import traceback
        traceback.print_exc()
    
    # 也测试其他点
    print(f'\n测试其他点:')
    test_points = [
        (0.0, 0.5),
        (0.5, 0.5),
        (0.99, 0.5),
    ]
    
    for u, v in test_points:
        N, id_vals = nrbbasisfun(np.array([[u], [v]]), NURBS)
        print(f'  ({u}, {v}): N非零={np.count_nonzero(N)}, sum={np.sum(N):.6f}')

if __name__ == '__main__':
    test_real_nurbs()

