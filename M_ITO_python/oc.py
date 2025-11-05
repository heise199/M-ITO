"""
优化准则模块 - 使用优化准则方法更新设计变量
"""
import numpy as np


def oc(X, R, Vmax, Sh, Hs, dJ_dp, dv_dp):
    """
    使用优化准则方法更新设计变量
    
    参数:
        X: 设计变量
        R: 基函数矩阵
        Vmax: 最大体积分数
        Sh: 过滤矩阵
        Hs: 行和向量
        dJ_dp: 目标函数对控制点密度的导数
        dv_dp: 体积对控制点密度的导数
    
    返回:
        X: 更新后的设计变量
    """
    l1 = 0
    l2 = 1e9
    move = 0.2
    
    # 避免除零错误
    while (l2 - l1) / max(l1 + l2, 1e-10) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        
        # 优化准则更新公式
        # 避免除零和负数开方
        dv_dp_safe = np.where(np.abs(dv_dp) < 1e-10, 1e-10, dv_dp)
        ratio = -dJ_dp / dv_dp_safe / max(lmid, 1e-10)
        ratio = np.maximum(ratio, 0)  # 确保非负，避免负数开方
        
        X_new = np.maximum(0, np.maximum(X['CtrPts'] - move, 
                                         np.minimum(1, np.minimum(X['CtrPts'] + move,
                                                                  X['CtrPts'] * np.sqrt(ratio)))))
        
        # 应用过滤
        X_new = (Sh @ X_new) / Hs
        
        # 更新高斯点密度
        X['GauPts'] = (R @ X_new).flatten()
        
        # 二分法调整拉格朗日乘子
        if np.mean(X['GauPts']) > Vmax:
            l1 = lmid
        else:
            l2 = lmid
    
    X['CtrPts_new'] = X_new
    return X

