"""
优化准则 (Optimality Criteria - OC)
从 MATLAB 代码转换
"""

import numpy as np


def oc(X, R, Vmax, Sh, Hs, dJ_dp, dv_dp):
    """
    优化准则方法更新设计变量
    
    参数:
        X: 设计变量字典
           - CtrPts: 控制点设计变量
        R: 从高斯点到控制点的映射矩阵
        Vmax: 最大体积分数
        Sh: Shepard 函数矩阵
        Hs: Shepard 函数行和
        dJ_dp: 目标函数关于设计变量的导数
        dv_dp: 体积关于设计变量的导数
    
    返回:
        X: 更新后的设计变量字典
           - CtrPts_new: 新的控制点设计变量
           - GauPts: 高斯点处的设计变量
    
    改编自 MATLAB IgaTop2D 代码
    """
    l1 = 0.0
    l2 = 1e9
    move = 0.2
    
    # 数值保护：避免除零
    dv_dp_safe = np.where(np.abs(dv_dp) < 1e-10, 1e-10, dv_dp)
    
    # 迭代次数限制
    max_iter = 1000
    iter_count = 0
    
    while iter_count < max_iter:
        # 防止除零
        if abs(l1 + l2) < 1e-20:
            break
            
        if (l2 - l1) / (l1 + l2) <= 1e-3:
            break
            
        lmid = 0.5 * (l2 + l1)
        
        # 限制 lmid 范围，避免极端值
        lmid = max(1e-10, min(lmid, 1e10))
        
        # OC 更新公式，添加数值保护
        # 计算 ratio = -dJ_dp / (dv_dp * lmid)
        ratio = -dJ_dp / (dv_dp_safe * lmid)
        
        # 确保 ratio 非负，才能开方
        ratio = np.maximum(ratio, 1e-10)
        
        # 计算更新
        X_CtrPts_new = np.maximum(0.0, 
                                   np.maximum(X['CtrPts'] - move,
                                             np.minimum(1.0,
                                                       np.minimum(X['CtrPts'] + move,
                                                                 X['CtrPts'] * np.sqrt(ratio)))))
        
        # 应用 Shepard 平滑
        X_CtrPts_new = (Sh @ X_CtrPts_new) / Hs.reshape(-1, 1)
        
        # 映射到高斯点
        X_GauPts = R @ X_CtrPts_new
        
        # 检查体积约束
        if np.mean(X_GauPts) > Vmax:
            l1 = lmid
        else:
            l2 = lmid
        
        iter_count += 1
    
    X['CtrPts_new'] = X_CtrPts_new
    X['GauPts'] = X_GauPts
    
    return X

# ======================================================================================================================
# 函数: oc
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

