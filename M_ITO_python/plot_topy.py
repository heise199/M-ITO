"""
拓扑优化结果绘图 (Topology Optimization Plotting)
从 MATLAB 代码转换
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置MATLAB风格的绘图参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.linewidth'] = 1.0


def plot_topy(X, GauPts, CtrPts, DenFied, Pos, case_number=None, save_dir='results'):
    """
    绘制拓扑优化结果并保存到文件
    
    参数:
        X: 设计变量字典
        GauPts: 高斯点信息
        CtrPts: 控制点信息
        DenFied: 密度场信息
        Pos: 图形位置 (在 matplotlib 中不使用)
        case_number: 案例编号，用于创建保存文件夹
        save_dir: 保存结果的根目录
    
    返回:
        X: 更新后的设计变量字典 (包含 DDF)
    
    改编自 MATLAB IgaTop2D 代码
    """
    # 创建保存目录
    if case_number is not None:
        output_dir = os.path.join(save_dir, f'case_{case_number}')
    else:
        output_dir = save_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.ioff()  # 关闭交互模式，不显示图像
    
    # Get coordinate ranges from DenFied, handling NaN/Inf values
    Ux_valid = DenFied['Ux'][np.isfinite(DenFied['Ux'])]
    Uy_valid = DenFied['Uy'][np.isfinite(DenFied['Uy'])]
    
    if len(Ux_valid) > 0:
        x_min, x_max = np.min(Ux_valid), np.max(Ux_valid)
    else:
        # Fallback to control points if DenFied has no valid values
        x_min, x_max = np.min(CtrPts['Cordis'][0, :]), np.max(CtrPts['Cordis'][0, :])
    
    if len(Uy_valid) > 0:
        y_min, y_max = np.min(Uy_valid), np.max(Uy_valid)
    else:
        # Fallback to control points if DenFied has no valid values
        y_min, y_max = np.min(CtrPts['Cordis'][1, :]), np.max(CtrPts['Cordis'][1, :])
    
    # Ensure valid finite values
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        x_min, x_max = 0.0, 1.0
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = 0.0, 1.0
    
    # Figure 1: Control points density 3D plot
    fig1 = plt.figure(1, figsize=(10, 7))
    plt.clf()
    ax1 = fig1.add_subplot(111, projection='3d')
    surf1 = ax1.scatter(CtrPts['Cordis'][0, :], CtrPts['Cordis'][1, :], X['CtrPts'].flatten(), 
                       c=X['CtrPts'].flatten(), cmap='jet', marker='o', s=30, edgecolors='k', linewidth=0.3)
    ax1.set_xlabel('X', fontsize=12, labelpad=10)
    ax1.set_ylabel('Y', fontsize=12, labelpad=10)
    ax1.set_zlabel('Density', fontsize=12, labelpad=10)
    ax1.set_title('Control Points Density', fontsize=14, pad=20)
    ax1.view_init(elev=30, azim=45)
    # 设置坐标轴范围
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.set_zlim([0, 1])
    fig1.colorbar(surf1, shrink=0.6, aspect=10)
    fig1.savefig(os.path.join(output_dir, 'control_points_density_3d.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 图2: 高斯点密度 3D 图
    fig2 = plt.figure(2, figsize=(10, 7))
    plt.clf()
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.scatter(GauPts['PCor'][0, :], GauPts['PCor'][1, :], X['GauPts'].flatten(), 
                       c=X['GauPts'].flatten(), cmap='jet', marker='o', s=15, alpha=0.6)
    ax2.set_xlabel('X', fontsize=12, labelpad=10)
    ax2.set_ylabel('Y', fontsize=12, labelpad=10)
    ax2.set_zlabel('Density', fontsize=12, labelpad=10)
    ax2.set_title('Gauss Points Density', fontsize=14, pad=20)
    ax2.view_init(elev=30, azim=45)
    # 设置坐标轴范围
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.set_zlim([0, 1])
    fig2.colorbar(surf2, shrink=0.6, aspect=10)
    fig2.savefig(os.path.join(output_dir, 'gauss_points_density_3d.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 计算密度分布函数 (DDF)
    # MATLAB: X.DDF = sum(DenFied.N.*X.CtrPts(DenFied.id),2);
    # MATLAB: X.DDF = reshape(X.DDF,numel(DenFied.U),numel(DenFied.V))';
    # 确保索引的形状正确
    # DenFied['id'] 的形状应该是 (npts, order)，其中 npts = len(U) * len(V)
    # X['CtrPts'] 的形状应该是 (n_ctrpts, 1)
    idx = (DenFied['id'] - 1).astype(int)  # 转为 0-based 索引
    
    # 边界检查：确保索引在有效范围内
    idx = np.clip(idx, 0, len(X['CtrPts']) - 1)
    
    # X['CtrPts'][idx] 会得到 (npts, order, 1)，需要 squeeze 掉最后一维
    ctrpts_vals = X['CtrPts'][idx].squeeze(-1)  # 现在是 (npts, order)
    
    # MATLAB的sum(..., 2)是按行求和（第2维），对应Python的axis=1
    X['DDF'] = np.sum(DenFied['N'] * ctrpts_vals, axis=1)
    
    # MATLAB: reshape(X.DDF,numel(DenFied.U),numel(DenFied.V))'
    # reshape成(len(U), len(V))，然后转置
    X['DDF'] = X['DDF'].reshape(len(DenFied['U']), len(DenFied['V'])).T
    
    # 图3: 密度分布函数 3D 曲面
    fig3 = plt.figure(3, figsize=(10, 7))
    plt.clf()
    ax3 = fig3.add_subplot(111, projection='3d')
    surf = ax3.plot_surface(DenFied['Ux'], DenFied['Uy'], X['DDF'], 
                            cmap='jet', alpha=0.9, edgecolor='none', antialiased=True)
    ax3.set_xlabel('X', fontsize=12, labelpad=10)
    ax3.set_ylabel('Y', fontsize=12, labelpad=10)
    ax3.set_zlabel('Density', fontsize=12, labelpad=10)
    ax3.set_title('Density Distribution Function', fontsize=14, pad=20)
    ax3.view_init(elev=30, azim=225)
    # 设置坐标轴范围与实际几何尺寸一致
    ax3.set_xlim([x_min, x_max])
    ax3.set_ylim([y_min, y_max])
    ax3.set_zlim([0, 1])
    fig3.colorbar(surf, shrink=0.6, aspect=10)
    fig3.savefig(os.path.join(output_dir, 'density_distribution_function_3d.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 图4: 高斯点密度 >= 0.5 的点
    fig4 = plt.figure(4, figsize=(10, 6))
    plt.clf()
    GauPts_PCor = GauPts['PCor'][0:2, X['GauPts'].flatten() >= 0.5]
    plt.plot(GauPts_PCor[0, :], GauPts_PCor[1, :], 'o', color='#000080', markersize=2, alpha=0.8)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Structural Topology (Scatter)', fontsize=14)
    plt.axis('equal')
    plt.grid(False)
    plt.box(True)
    fig4.savefig(os.path.join(output_dir, 'gauss_points_scatter.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig4)
    
    # 图5: 结构拓扑 (等高线图)
    # MATLAB: contourf(DenFied.Ux, DenFied.Uy, X.DDF, [0.5 0.5], ...)
    # MATLAB的[0.5 0.5]表示绘制密度>=0.5的区域
    fig5 = plt.figure(5, figsize=(10, 6))
    plt.clf()
    # 使用levels=[0.5]绘制密度>=0.5的区域，或者使用levels=[0.5, 1.0]绘制0.5到1.0之间的区域
    # 为了匹配MATLAB的行为，我们使用levels=[0.5]并设置extend='max'来填充>=0.5的区域
    contour = plt.contourf(DenFied['Ux'], DenFied['Uy'], X['DDF'], levels=[0.5, 1.0], 
                           colors=['#000080'], alpha=0.9, extend='both')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Structural Topology (Contour)', fontsize=14)
    plt.axis('equal')
    plt.grid(False)
    plt.box(True)
    fig5.savefig(os.path.join(output_dir, 'structural_topology_contour.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig5)
    
    # 清理所有图形以释放内存
    plt.close('all')
    
    return X


# ======================================================================================================================
# 函数: plot_topy
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

