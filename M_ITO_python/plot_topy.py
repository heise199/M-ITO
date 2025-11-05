"""
拓扑优化结果绘图模块
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 开启交互模式，允许图形实时更新
plt.ion()


def plot_topy(X, GauPts, CtrPts, DenFied, Pos):
    """
    绘制拓扑优化结果
    
    参数:
        X: 设计变量
        GauPts: 高斯点信息
        CtrPts: 控制点信息
        DenFied: 密度场数据
        Pos: 图形位置信息
    
    返回:
        X: 设计变量（可能包含DDF字段）
    """
    # 图1: 控制点密度（三维散点，与MATLAB一致）
    fig1 = plt.figure(1)
    fig1.clf()
    ax1 = fig1.add_subplot(111, projection='3d')
    sc1 = ax1.scatter(CtrPts['Cordis'][0, :], CtrPts['Cordis'][1, :], X['CtrPts'],
                      c=X['CtrPts'], cmap='jet', s=10, vmin=0, vmax=1)
    try:
        ax1.view_init(elev=30, azim=-60)
    except Exception:
        pass
    ax1.set_title('Control Points Density')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Density')
    fig1.colorbar(sc1, ax=ax1, shrink=0.6, pad=0.1)
    fig1.show()
    
    # 图2: 高斯点密度（三维散点，与MATLAB一致）
    fig2 = plt.figure(2)
    fig2.clf()
    ax2 = fig2.add_subplot(111, projection='3d')
    sc2 = ax2.scatter(GauPts['PCor'][0, :], GauPts['PCor'][1, :], X['GauPts'],
                      c=X['GauPts'], cmap='jet', s=8, vmin=0, vmax=1)
    try:
        ax2.view_init(elev=30, azim=-60)
    except Exception:
        pass
    ax2.set_title('Gauss Points Density')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Density')
    fig2.colorbar(sc2, ax=ax2, shrink=0.6, pad=0.1)
    fig2.show()
    
    # 计算密度分布函数(DDF)
    # MATLAB: X.DDF = sum(DenFied.N.*X.CtrPts(DenFied.id),2);
    # MATLAB: X.DDF = reshape(X.DDF,numel(DenFied.U),numel(DenFied.V))';
    # 稀疏矩阵方式：与构建 R 相同，数值更稳定
    X['DDF'] = np.asarray(DenFied['R'] @ X['CtrPts']).ravel()
    
    # MATLAB: reshape(X.DDF,numel(DenFied.U),numel(DenFied.V))'
    # 注意：MATLAB的reshape是按列填充，然后转置
    X['DDF'] = X['DDF'].reshape(len(DenFied['U']), len(DenFied['V'])).T
    
    # 图3: 密度分布函数(DDF) - 三维曲面（与MATLAB一致）
    fig3 = plt.figure(3)
    fig3.clf()
    ax3 = fig3.add_subplot(111, projection='3d')
    surf = ax3.plot_surface(DenFied['Ux'], DenFied['Uy'], X['DDF'],
                           cmap='jet', alpha=0.8, linewidth=0, antialiased=True)
    try:
        ax3.view_init(elev=30, azim=-60)
    except Exception:
        pass
    ax3.set_title('Density Distribution Function')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Density')
    fig3.colorbar(surf, ax=ax3, shrink=0.6, pad=0.1)
    fig3.show()
    
    # MATLAB: GauPts_PCor = GauPts.PCor(1:2, X.GauPts>=0.5);
    # MATLAB: plot(GauPts_PCor(1,:),GauPts_PCor(2,:),'.','color',[0.5 0 0.8]);
    # 图4: 结构拓扑（高斯点）- 二维散点图
    fig4 = plt.figure(4)
    fig4.clf()
    ax4 = fig4.add_subplot(111)
    mask = X['GauPts'] >= 0.5
    GauPts_PCor = GauPts['PCor'][:2, mask]
    ax4.scatter(GauPts_PCor[0, :], GauPts_PCor[1, :], 
               color=[0.5, 0, 0.8], s=20)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title('Structural Topology (Gauss Points)')
    fig4.show()
    
    # MATLAB: contourf(DenFied.Ux, DenFied.Uy, X.DDF, [0.5 0.5], 'facecolor', [0.5 0 0.8], 'edgecolor', [1 1 1]);
    # 图5: 结构拓扑（等高线）- 二维等高线图
    fig5 = plt.figure(5)
    fig5.clf()
    ax5 = fig5.add_subplot(111)
    # 严格模仿MATLAB：contourf(...,[0.5 0.5],...) 的等效实现（Matplotlib要求递增levels）
    zmin = float(np.nanmin(X['DDF'])) if np.size(X['DDF']) else 0.0
    zmax = float(np.nanmax(X['DDF'])) if np.size(X['DDF']) else 1.0
    if zmax <= 0.5:
        levels = [zmax - 1e-9, 0.5]
    elif zmin >= 0.5:
        levels = [0.5, zmin + 1e-9]
    else:
        levels = [0.5, zmax + 1e-9]
    contour = ax5.contourf(DenFied['Ux'], DenFied['Uy'], X['DDF'],
                           levels=levels, colors=[[0.5, 0, 0.8]], antialiased=True)
    ax5.set_aspect('equal')
    ax5.axis('off')
    ax5.set_title('Structural Topology (Contour)')
    fig5.show()
    
    # 强制刷新所有图形窗口
    plt.draw()
    plt.pause(0.1)  # 增加暂停时间以确保图形更新
    
    return X

