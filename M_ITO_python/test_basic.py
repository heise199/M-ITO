"""
基础功能测试脚本
"""

import numpy as np
import sys

def test_imports():
    """测试所有模块能否正常导入"""
    print("测试模块导入...")
    try:
        # NURBS 工具箱
        from nurbs import findspan, basisfun, basisfunder, numbasisfun
        from nurbs import nrbmak, nrbeval, nrbbasisfun, nrbbasisfunder
        from nurbs import nrbdegelev, nrbkntins
        print("  ✓ NURBS 工具箱导入成功")
        
        # 主要模块
        from guadrature import guadrature
        from geom_mod import geom_mod
        from pre_iga import pre_iga
        from boun_cond import boun_cond
        from shep_fun import shep_fun
        from stiff_ele2d import stiff_ele2d
        from stiff_ass2d import stiff_ass2d
        from solving import solving
        from oc import oc
        from plot_data import plot_data
        from plot_topy import plot_topy
        from iga_top2d import iga_top2d
        print("  ✓ 所有主要模块导入成功")
        
        return True
    except Exception as e:
        print(f"  ✗ 导入失败: {str(e)}")
        return False


def test_nurbs_basic():
    """测试 NURBS 基础函数"""
    print("\n测试 NURBS 基础函数...")
    try:
        from nurbs import nrbmak, nrbeval
        
        # 创建简单的 NURBS 曲面
        knots = [[0, 0, 1, 1], [0, 0, 1, 1]]
        ControlPts = np.zeros((4, 2, 2))
        ControlPts[:, :, 0] = np.array([[0, 1], [0, 0], [0, 0], [1, 1]])
        ControlPts[:, :, 1] = np.array([[0, 1], [1, 1], [0, 0], [1, 1]])
        
        coefs = ControlPts.copy()
        coefs[0, :, :] = ControlPts[0, :, :] * ControlPts[3, :, :]
        coefs[1, :, :] = ControlPts[1, :, :] * ControlPts[3, :, :]
        coefs[2, :, :] = ControlPts[2, :, :] * ControlPts[3, :, :]
        coefs[3, :, :] = ControlPts[3, :, :]
        
        NURBS = nrbmak(coefs, knots)
        print(f"  ✓ NURBS 创建成功: number={NURBS['number']}, order={NURBS['order']}")
        
        # 测试求值
        p, w = nrbeval(NURBS, np.array([[0.5], [0.5]]))
        print(f"  ✓ NURBS 求值成功: p.shape={p.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ NURBS 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_geometry_model():
    """测试几何模型生成"""
    print("\n测试几何模型生成...")
    try:
        from geom_mod import geom_mod
        
        # 测试悬臂梁几何
        NURBS = geom_mod(L=10, W=5, Order=[0, 0], Num=[11, 6], BoundCon=1)
        print(f"  ✓ 几何模型创建成功: number={NURBS['number']}")
        
        return True
    except Exception as e:
        print(f"  ✗ 几何模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_guadrature():
    """测试高斯积分"""
    print("\n测试高斯积分...")
    try:
        from guadrature import guadrature
        
        weight, points = guadrature(3, 2)
        print(f"  ✓ 高斯积分生成成功: {len(weight)} 个积分点")
        print(f"    权重和 = {np.sum(weight):.6f} (应该接近 4.0)")
        
        return True
    except Exception as e:
        print(f"  ✗ 高斯积分测试失败: {str(e)}")
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("IGA 拓扑优化 Python 实现 - 基础功能测试")
    print("="*60)
    
    all_pass = True
    
    # 测试导入
    if not test_imports():
        all_pass = False
        print("\n导入测试失败，停止后续测试")
        sys.exit(1)
    
    # 测试 NURBS
    if not test_nurbs_basic():
        all_pass = False
    
    # 测试几何模型
    if not test_geometry_model():
        all_pass = False
    
    # 测试高斯积分
    if not test_guadrature():
        all_pass = False
    
    print("\n" + "="*60)
    if all_pass:
        print("✓ 所有基础测试通过！")
        print("\n建议:")
        print("1. 运行 'python CASE.py' 选择案例 6 (测试案例) 进行完整测试")
        print("2. 如果测试案例运行成功，再尝试其他更大规模的案例")
    else:
        print("✗ 部分测试失败，请检查错误信息")
    print("="*60)


if __name__ == '__main__':
    main()

