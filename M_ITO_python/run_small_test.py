"""
运行小规模测试案例
"""

from iga_top2d import iga_top2d
import sys

def main():
    """
    运行小规模测试案例
    """
    print('\n' + '='*80)
    print('运行小规模测试案例 - Python 实现')
    print('='*80 + '\n')
    
    try:
        # 使用小规模参数进行快速测试
        L = 10
        W = 5
        Order = [1, 1]
        Num = [21, 11]  # 中等规模
        BoundCon = 1  # 悬臂梁
        Vmax = 0.2
        penal = 3
        rmin = 2
        
        print(f'测试参数:')
        print(f'  几何: L={L}, W={W}')
        print(f'  NURBS: Order={Order}, Num={Num}')
        print(f'  边界条件: {BoundCon} (悬臂梁)')
        print(f'  优化参数: Vmax={Vmax}, penal={penal}, rmin={rmin}')
        print()
        
        # 运行优化
        X, Data, Iter_Ch = iga_top2d(L, W, Order, Num, BoundCon, Vmax, penal, rmin, case_number=1)
        
        print('\n' + '='*80)
        print('测试完成!')
        print('='*80)
        
        # 显示结果摘要
        import numpy as np
        valid_data = Data[Data[:, 0] > 0]
        if len(valid_data) > 0:
            print(f'\n结果摘要:')
            print(f'  总迭代次数: {len(valid_data)}')
            print(f'  初始目标函数: {valid_data[0, 0]:.6f}')
            print(f'  最终目标函数: {valid_data[-1, 0]:.6f}')
            print(f'  目标函数改善: {(valid_data[0, 0] - valid_data[-1, 0]) / valid_data[0, 0] * 100:.2f}%')
            print(f'  最终体积分数: {valid_data[-1, 1]:.3f}')
            print(f'  目标体积分数: {Vmax:.3f}')
        
        return X, Data, Iter_Ch
        
    except Exception as e:
        print(f'\n错误: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    X, Data, Iter_Ch = main()

