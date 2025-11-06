"""
测试载荷计算
"""

import numpy as np
from geom_mod import geom_mod
from pre_iga import pre_iga
from boun_cond import boun_cond

def test_load():
    # 使用小规模参数
    L = 10
    W = 5
    Order = [1, 1]
    Num = [11, 6]
    BoundCon = 1
    
    print('生成几何模型...')
    NURBS = geom_mod(L, W, Order, Num, BoundCon)
    print(f'NURBS.knots[0]: [{NURBS["knots"][0][0]}, ..., {NURBS["knots"][0][-1]}]')
    print(f'NURBS.knots[1]: [{NURBS["knots"][1][0]}, ..., {NURBS["knots"][1][-1]}]')
    print(f'NURBS.number: {NURBS["number"]}')
    
    print('\nIGA 预处理...')
    CtrPts, Ele, GauPts = pre_iga(NURBS)
    Dim = len(NURBS['order'])
    Dofs_Num = Dim * CtrPts['Num']
    
    print(f'CtrPts.Num = {CtrPts["Num"]}')
    print(f'Dofs.Num = {Dofs_Num}')
    
    print('\n设置边界条件...')
    DBoudary, F = boun_cond(CtrPts, BoundCon, NURBS, Dofs_Num)
    
    print(f'\n载荷向量检查:')
    print(f'  F非零数量: {np.count_nonzero(F)}')
    print(f'  F总和: {np.sum(np.abs(F)):.6e}')
    print(f'  F最大值: {np.max(np.abs(F)):.6e}')
    
    if np.count_nonzero(F) == 0:
        print('\n错误：载荷向量为零！')
        return False
    else:
        print('\n成功：载荷向量非零')
        return True

if __name__ == '__main__':
    success = test_load()
    if not success:
        exit(1)

