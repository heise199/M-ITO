"""
对比测试脚本 - 用于验证MATLAB和Python的一致性
"""

import numpy as np
from geom_mod import geom_mod
from pre_iga import pre_iga
from nurbs import nrbbasisfun, nrbbasisfunder
import sys

def test_small_case():
    """
    测试小规模案例
    """
    print('='*80)
    print('测试小规模案例')
    print('='*80)
    
    # 使用小规模参数
    L = 10
    W = 5
    Order = [1, 1]
    Num = [11, 6]  # 小网格
    BoundCon = 1
    
    print(f'\n参数: L={L}, W={W}, Order={Order}, Num={Num}')
    
    # 生成几何模型
    print('\n生成几何模型...')
    NURBS = geom_mod(L, W, Order, Num, BoundCon)
    print(f'NURBS.number = {NURBS["number"]}')
    print(f'NURBS.order = {NURBS["order"]}')
    print(f'NURBS.knots[0] (前10个): {NURBS["knots"][0][:10]}')
    print(f'NURBS.knots[1] (前10个): {NURBS["knots"][1][:10]}')
    
    # IGA 预处理
    print('\nIGA 预处理...')
    CtrPts, Ele, GauPts = pre_iga(NURBS)
    print(f'CtrPts.Num = {CtrPts["Num"]}')
    print(f'Ele.Num = {Ele["Num"]}, Ele.NumU = {Ele["NumU"]}, Ele.NumV = {Ele["NumV"]}')
    print(f'Ele.CtrPtsNum = {Ele["CtrPtsNum"]}')
    print(f'GauPts.Num = {GauPts["Num"]}')
    
    # 检查单元序列
    print(f'\nEle.Seque shape: {Ele["Seque"].shape}')
    print(f'Ele.Seque (前3行3列):\n{Ele["Seque"][:3, :3]}')
    
    # 检查控制点序列
    print(f'\nCtrPts.Seque shape: {CtrPts["Seque"].shape}')
    print(f'CtrPts.Seque (前5行5列):\n{CtrPts["Seque"][:5, :5]}')
    
    # 检查单元控制点连接
    print(f'\nEle.CtrPtsCon shape: {Ele["CtrPtsCon"].shape}')
    print(f'Ele.CtrPtsCon (前3个单元):')
    for i in range(min(3, Ele['Num'])):
        print(f'  单元 {i+1}: {Ele["CtrPtsCon"][i, :]}')
    
    # 准备高斯点坐标
    print('\n准备高斯点坐标...')
    GauPts['Cor'] = np.vstack([
        GauPts['CorU'].T.flatten(order='F'),
        GauPts['CorV'].T.flatten(order='F')
    ])
    print(f'GauPts.Cor shape: {GauPts["Cor"].shape}')
    print(f'GauPts.Cor (前5列):\n{GauPts["Cor"][:, :5]}')
    
    # 计算基函数
    print('\n计算基函数...')
    N, id_vals = nrbbasisfun(GauPts['Cor'], NURBS)
    print(f'N shape: {N.shape}')
    print(f'id_vals shape: {id_vals.shape}')
    print(f'N (前3行):')
    for i in range(min(3, N.shape[0])):
        print(f'  高斯点 {i+1}: {N[i, :]}')
    print(f'id_vals (前3行):')
    for i in range(min(3, id_vals.shape[0])):
        print(f'  高斯点 {i+1}: {id_vals[i, :]}')
    
    # 计算基函数导数
    print('\n计算基函数导数...')
    dRu, dRv, id_dR = nrbbasisfunder(GauPts['Cor'], NURBS)
    print(f'dRu shape: {dRu.shape}')
    print(f'dRv shape: {dRv.shape}')
    print(f'id_dR shape: {id_dR.shape}')
    
    # 检查第一个单元的高斯点
    print('\n检查第一个单元 (ide=1) 的高斯点:')
    ide = 0  # Python 0-based
    print(f'GauPts.Seque[{ide}, :] = {GauPts["Seque"][ide, :]} (1-based)')
    
    for i in range(Ele['GauPtsNum']):
        GptOrder = GauPts['Seque'][ide, i] - 1  # 转为 0-based
        print(f'\n  高斯点 {i+1}, GptOrder = {GptOrder+1} (1-based):')
        print(f'    id_vals[{GptOrder}] = {id_vals[GptOrder, :]}')
        print(f'    id_dR[{GptOrder}] = {id_dR[GptOrder, :]}')
        print(f'    dRu[{GptOrder}] = {dRu[GptOrder, :]}')
        print(f'    dRv[{GptOrder}] = {dRv[GptOrder, :]}')
    
    print(f'\n  单元 1 的控制点编号: {Ele["CtrPtsCon"][ide, :]}')
    
    # 检查id_vals和id_dR是否相同
    print('\n检查 id_vals 和 id_dR 是否相同:')
    if np.allclose(id_vals, id_dR):
        print('  ✓ id_vals 和 id_dR 相同')
    else:
        print('  ✗ id_vals 和 id_dR 不同!')
        diff_count = np.sum(id_vals != id_dR)
        print(f'    不同元素数量: {diff_count} / {id_vals.size}')
    
    # 检查单元内的高斯点是否有相同的id
    print('\n检查单元内的高斯点是否有相同的基函数编号:')
    for ide in range(min(3, Ele['Num'])):
        gp_ids = []
        for i in range(Ele['GauPtsNum']):
            GptOrder = GauPts['Seque'][ide, i] - 1
            gp_ids.append(id_dR[GptOrder, :])
        gp_ids = np.array(gp_ids)
        
        # 检查是否所有行相同
        all_same = np.all(gp_ids == gp_ids[0, :], axis=0)
        if np.all(all_same):
            print(f'  单元 {ide+1}: ✓ 所有高斯点有相同的基函数编号')
            print(f'    编号: {gp_ids[0, :]}')
            print(f'    Ele.CtrPtsCon[{ide}]: {Ele["CtrPtsCon"][ide, :]}')
            if np.array_equal(np.sort(gp_ids[0, :]), np.sort(Ele["CtrPtsCon"][ide, :])):
                print(f'    ✓ 与 Ele.CtrPtsCon 匹配')
            else:
                print(f'    ✗ 与 Ele.CtrPtsCon 不匹配!')
        else:
            print(f'  单元 {ide+1}: ✗ 高斯点有不同的基函数编号!')
            for i in range(Ele['GauPtsNum']):
                print(f'    高斯点 {i+1}: {gp_ids[i, :]}')
    
    print('\n' + '='*80)
    print('测试完成')
    print('='*80)

if __name__ == '__main__':
    test_small_case()

