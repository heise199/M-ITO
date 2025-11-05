"""
绘图数据准备 (Plot Data Preparation)
从 MATLAB 代码转换
"""

import numpy as np
from nurbs import nrbbasisfun, nrbeval


def plot_data(Num, NURBS):
    """
    准备绘图所需的密度场数据
    
    参数:
        Num: 节点数量 [Num_U, Num_V]
        NURBS: NURBS 几何结构
    
    返回:
        DenFied: 密度场信息字典
        Pos: 图形位置信息字典 (此处简化，不考虑屏幕尺寸)
    
    改编自 MATLAB IgaTop2D 代码
    """
    DenFied = {}
    Pos = {}
    
    # 简化的位置设置 (Python/matplotlib 不需要像 MATLAB 那样精确的像素定位)
    Pos['p1'] = [0.1, 0.6, 0.35, 0.35]  # [left, bottom, width, height]
    Pos['p2'] = [0.55, 0.6, 0.35, 0.35]
    Pos['p3'] = [0.1, 0.1, 0.35, 0.35]
    Pos['p4'] = [0.55, 0.1, 0.35, 0.35]
    
    # 生成密集的求值网格
    Uknots = np.linspace(0, 1, 10 * Num[0])
    Vknots = np.linspace(0, 1, 10 * Num[1])
    
    # 计算基函数
    N_f, id_f = nrbbasisfun([Uknots, Vknots], NURBS)
    
    # 在网格上求值 NURBS
    PCor_U, PCor_W = nrbeval(NURBS, [Uknots, Vknots])
    
    # 调试：检查返回值
    print(f'[Debug plot_data] PCor_U shape: {PCor_U.shape}')
    print(f'[Debug plot_data] Uknots: [{Uknots[0]:.2f}, {Uknots[-1]:.2f}], len={len(Uknots)}')
    print(f'[Debug plot_data] Vknots: [{Vknots[0]:.2f}, {Vknots[-1]:.2f}], len={len(Vknots)}')
    print(f'[Debug plot_data] PCor_U[0] 范围: [{np.min(PCor_U[0]):.2f}, {np.max(PCor_U[0]):.2f}]')
    print(f'[Debug plot_data] PCor_U[1] 范围: [{np.min(PCor_U[1]):.2f}, {np.max(PCor_U[1]):.2f}]')
    
    # 提取 X 和 Y 坐标
    PCor_Ux = PCor_U[0, :, :].T
    PCor_Uy = PCor_U[1, :, :].T
    
    print(f'[Debug plot_data] PCor_Ux 范围: [{np.min(PCor_Ux):.2f}, {np.max(PCor_Ux):.2f}]')
    print(f'[Debug plot_data] PCor_Uy 范围: [{np.min(PCor_Uy):.2f}, {np.max(PCor_Uy):.2f}]\n')
    
    # 存储数据
    DenFied['N'] = N_f
    DenFied['id'] = id_f
    DenFied['U'] = Uknots
    DenFied['V'] = Vknots
    DenFied['Ux'] = PCor_Ux
    DenFied['Uy'] = PCor_Uy
    
    return DenFied, Pos


# ======================================================================================================================
# 函数: plot_data
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

