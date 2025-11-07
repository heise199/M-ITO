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
    
    # Evaluate NURBS on grid
    PCor_U, PCor_W = nrbeval(NURBS, [Uknots, Vknots])
    # MATLAB: PCor_U = PCor_U./PCor_W;
    # IMPORTANT: Python's nrbeval already divides by weights and returns Cartesian coordinates
    # MATLAB's nrbeval returns homogeneous coordinates (wx, wy, wz), so MATLAB code divides by weights
    # But Python's nrbeval implementation already does this division internally
    # Therefore, we should NOT divide again - PCor_U is already Cartesian coordinates
    
    # Extract X and Y coordinates
    # MATLAB: PCor_Ux = reshape(PCor_U(1,:),numel(Uknots),numel(Vknots))';
    # In MATLAB, when nrbeval is called with {Uknots, Vknots}, it returns PCor_U
    # The exact shape depends on MATLAB's nrbeval implementation, but typically:
    # PCor_U(1,:) extracts first coordinate (X) as a row vector, column-major flattened
    # reshape(..., nt1, nt2) reshapes column-wise to (nt1, nt2), then transpose to (nt2, nt1)
    nt1 = len(Uknots)
    nt2 = len(Vknots)
    
    if len(PCor_U.shape) == 3:
        # PCor_U is (3, nt1, nt2) from Python nrbeval
        # MATLAB's PCor_U(1,:) on a (3, nt1, nt2) array extracts first row, column-major: (1, nt1*nt2)
        # We need to flatten column-wise, reshape column-wise, then transpose
        PCor_Ux_flat = PCor_U[0, :, :].flatten(order='F')  # Column-major flatten: (nt1*nt2,)
        PCor_Ux = PCor_Ux_flat.reshape(nt1, nt2, order='F').T  # Reshape column-wise, then transpose: (nt2, nt1)
        
        PCor_Uy_flat = PCor_U[1, :, :].flatten(order='F')  # Column-major flatten
        PCor_Uy = PCor_Uy_flat.reshape(nt1, nt2, order='F').T  # Reshape column-wise, then transpose
    else:
        # If PCor_U is already flattened to (3, nt1*nt2)
        PCor_U_flat = PCor_U.reshape(3, -1, order='F')
        PCor_Ux_flat = PCor_U_flat[0, :]  # (nt1*nt2,)
        PCor_Ux = PCor_Ux_flat.reshape(nt1, nt2, order='F').T  # Reshape column-wise then transpose
        PCor_Uy_flat = PCor_U_flat[1, :]
        PCor_Uy = PCor_Uy_flat.reshape(nt1, nt2, order='F').T
    
    # Store data
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

