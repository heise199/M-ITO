"""
案例运行脚本
从 MATLAB 代码转换
"""

from iga_top2d import iga_top2d


def run_case():
    """
    运行不同的拓扑优化案例
    
    改编自 MATLAB IgaTop2D 代码
    """
    print('请选择要运行的案例:')
    print('1: 悬臂梁 (Cantilever beam)')
    print('2: MBB 梁 (MBB beam)')
    print('3: Michell 型结构 (Michell-type structure)')
    print('4: L 梁 (L beam)')
    print('5: 四分之一环形 (Quarter annulus)')
    print('6: 测试案例 (Test case)')
    
    case_num = input('请输入案例编号 (1-6) [默认 1]: ')
    if not case_num:
        case_num = '1'
    
    case_num = int(case_num)
    
    if case_num == 1:
        print('\n运行案例 1: 悬臂梁')
        X, Data, Iter_Ch = iga_top2d(10, 5, [1, 1], [161, 81], 1, 0.2, 3, 2, case_number=1)
    
    elif case_num == 2:
        print('\n运行案例 2: MBB 梁')
        X, Data, Iter_Ch = iga_top2d(18, 3, [1, 1], [241, 41], 2, 0.2, 3, 2, case_number=2)
    
    elif case_num == 3:
        print('\n运行案例 3: Michell 型结构')
        X, Data, Iter_Ch = iga_top2d(10, 4, [1, 1], [101, 41], 3, 0.2, 3, 2, case_number=3)
    
    elif case_num == 4:
        print('\n运行案例 4: L 梁')
        X, Data, Iter_Ch = iga_top2d(10, 5, [1, 1], [101, 51], 4, 0.3, 3, 2, case_number=4)
    
    elif case_num == 5:
        print('\n运行案例 5: 四分之一环形')
        X, Data, Iter_Ch = iga_top2d(10, 10, [0, 1], [101, 51], 5, 0.4, 3, 2, case_number=5)
    
    elif case_num == 6:
        print('\n运行案例 6: 测试案例 (小规模)')
        X, Data, Iter_Ch = iga_top2d(10, 5, [0, 1], [21, 21], 5, 0.4, 3, 2, case_number=6)
    
    else:
        print('无效的案例编号，运行默认案例 1')
        X, Data, Iter_Ch = iga_top2d(10, 5, [1, 1], [161, 81], 1, 0.2, 3, 2, case_number=1)
    
    return X, Data, Iter_Ch


if __name__ == '__main__':
    try:
        X, Data, Iter_Ch = run_case()
        print('\n程序执行完毕!')
    except KeyboardInterrupt:
        print('\n\n程序被用户中断')
    except Exception as e:
        print(f'\n错误: {str(e)}')
        import traceback
        traceback.print_exc()


# ======================================================================================================================
# 脚本: CASE.py
#
# 用于等几何拓扑优化的紧凑高效 Python 实现
#
# 开发者: 原始 MATLAB 代码 - Jie Gao
# Email: JieGao@hust.edu.cn
# Python 转换: 2025
#
# 主要参考文献:
#
# (1) Jie Gao, Lin Wang, Zhen Luo, Liang Gao. IgaTop: an implementation of topology optimization for structures
# using IGA in Matlab. Structural and multidisciplinary optimization.
#
# (2) Jie Gao, Liang Gao, Zhen Luo, Peigen Li. Isogeometric topology optimization for continuum structures using
# density distribution function. Int J Numer Methods Eng, 2019, 119:991–1017
#
# *********************************************   免责声明   *******************************************************
# 作者保留程序的所有权利。程序可用于学术和教育目的。作者不保证代码没有错误，
# 并且不对因使用程序而引起的任何事件承担责任。
# ======================================================================================================================

