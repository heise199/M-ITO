"""
诊断优化过程问题
检查目标函数、灵敏度、OC更新等
"""
import sys
import numpy as np
sys.path.insert(0, 'src')

from src.geometry.geom_mod_2d import geom_mod_2d
from src.iga.pre_iga_2d import pre_iga_2d
from src.boundary import boun_cond
from src.iga import stiff_ele_2d, stiff_ass_2d
from src.solver import solving
from src.optimization import oc, shep_fun
from src.nurbs import nrbeval, nrbbasisfun, nrbbasisfunder
from scipy.sparse import csc_matrix

# 创建几何模型
L, W = 5, 2
Order = [1, 1]
Num = [51, 21]
BoundCon = 3
Vmax = 0.5
penal = 3.0
rmin = 1.5
Emin = 1e-6

print("=" * 80)
print("诊断优化过程")
print("=" * 80)

# 1. 几何和IGA预处理
print("\n1. 几何和IGA预处理...")
NURBS = geom_mod_2d(L, W, Order, Num, BoundCon)
CtrPts, Ele, GauPts = pre_iga_2d(NURBS)

# 2. 边界条件
print("2. 边界条件...")
DBoudary, Dofs, F = boun_cond(CtrPts, BoundCon, NURBS, CtrPts['Num'] * 2)
print(f"  总自由度数: {Dofs['Num']}")
print(f"  自由自由度数: {len(Dofs['Free'])}")
print(f"  固定自由度数: {len(Dofs['Fixed'])}")

# 3. 基函数和插值矩阵
print("3. 计算基函数和插值矩阵...")
GauPts_Cor = np.array([
    GauPts['CorU'].flatten(),
    GauPts['CorV'].flatten()
])
N, id_arr = nrbbasisfun(GauPts_Cor, NURBS)
dRu, dRv = nrbbasisfunder(GauPts_Cor, NURBS)

R = csc_matrix((GauPts['Num'], CtrPts['Num']))
for i in range(GauPts['Num']):
    valid_ids = id_arr[i, :][id_arr[i, :] >= 0]
    if len(valid_ids) > 0:
        R[i, valid_ids] = N[i, valid_ids]

# 4. 初始化设计变量
print("4. 初始化设计变量...")
X = {'CtrPts': np.ones(CtrPts['Num'])}
X['GauPts'] = R @ X['CtrPts']
print(f"  初始体积分数: {np.mean(X['GauPts']):.6f}")

# 5. Shepard过滤
print("5. Shepard过滤...")
Sh, Hs = shep_fun(CtrPts, rmin)

# 6. 材料属性
print("6. 材料属性...")
E = 1.0
nu = 0.3
DH = np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1-nu)/2]
]) * (E / (1 - nu**2))

# 7. 第一次迭代
print("\n7. 第一次迭代...")
print("  7.1 计算单元刚度矩阵...")
KE, dKE, dv_dg = stiff_ele_2d(X, penal, Emin, DH, CtrPts, Ele, GauPts, dRu, dRv)

print("  7.2 组装全局刚度矩阵...")
K = stiff_ass_2d(KE, CtrPts, Ele, 2, Dofs['Num'])

print("  7.3 求解位移...")
U = solving(CtrPts, DBoudary, Dofs, K, F, BoundCon)
print(f"    位移范围: [{np.min(U):.6e}, {np.max(U):.6e}]")
if np.any(np.isnan(U)) or np.any(np.isinf(U)):
    print("    ⚠ 警告：位移包含NaN或Inf！")

print("  7.4 计算目标函数...")
J = 0.0
dJ_dg = np.zeros(GauPts['Num'])

for ide in range(Ele['Num']):
    Ele_NoCtPt = Ele['CtrPtsCon'][ide, :] - 1
    valid_mask = Ele_NoCtPt >= 0
    Ele_NoCtPt = Ele_NoCtPt[valid_mask]
    
    if len(Ele_NoCtPt) == 0:
        continue
    
    edof = np.concatenate([
        Ele_NoCtPt,
        Ele_NoCtPt + CtrPts['Num']
    ])
    
    Ue = U[edof]
    Ke = KE[ide]
    
    if Ke is not None and len(Ue) == Ke.shape[0]:
        J += Ue.T @ Ke @ Ue
        
        for i in range(Ele['GauPtsNum']):
            GptOrder = GauPts['Seque'][ide, i] - 1
            if dKE[ide][i] is not None:
                dJ_dg[GptOrder] = -Ue.T @ dKE[ide][i] @ Ue

print(f"    目标函数值 J = {J:.6e}")
print(f"    灵敏度范围: [{np.min(dJ_dg):.6e}, {np.max(dJ_dg):.6e}]")

if np.any(np.isnan(J)) or np.any(np.isinf(J)):
    print("    ⚠ 警告：目标函数包含NaN或Inf！")

if np.any(np.isnan(dJ_dg)) or np.any(np.isinf(dJ_dg)):
    print("    ⚠ 警告：灵敏度包含NaN或Inf！")
    nan_count = np.sum(np.isnan(dJ_dg))
    print(f"    NaN数量: {nan_count}/{len(dJ_dg)}")

print("  7.5 灵敏度过滤...")
dJ_dp = R.T @ dJ_dg
dJ_dp = Sh @ (dJ_dp / (Hs + 1e-10))
dv_dp = R.T @ dv_dg
dv_dp = Sh @ (dv_dp / (Hs + 1e-10))

print(f"    过滤后灵敏度范围: [{np.min(dJ_dp):.6e}, {np.max(dJ_dp):.6e}]")
print(f"    体积灵敏度范围: [{np.min(dv_dp):.6e}, {np.max(dv_dp):.6e}]")

print("  7.6 OC更新...")
X_new = oc(X.copy(), R, Vmax, Sh, Hs, dJ_dp, dv_dp, use_gpu=False)

change = np.max(np.abs(X_new['CtrPts'] - X['CtrPts']))
print(f"    设计变量变化: {change:.6e}")
print(f"    新体积分数: {np.mean(X_new['GauPts']):.6f}")

# 诊断总结
print("\n" + "=" * 80)
print("诊断总结")
print("=" * 80)

issues = []

if np.any(np.isnan(U)) or np.any(np.isinf(U)):
    issues.append("❌ 位移包含NaN或Inf - 检查刚度矩阵和边界条件")

if np.any(np.isnan(J)) or np.any(np.isinf(J)):
    issues.append("❌ 目标函数包含NaN或Inf - 检查单元刚度矩阵")

if np.any(np.isnan(dJ_dg)) or np.any(np.isinf(dJ_dg)):
    issues.append("❌ 灵敏度包含NaN或Inf - 检查单元刚度矩阵导数")

if J <= 0:
    issues.append("⚠️ 目标函数值 <= 0 - 可能不正常")

if change < 1e-10:
    issues.append("⚠️ 设计变量变化太小 - OC可能没有更新")

if len(issues) == 0:
    print("✓ 未发现明显问题")
else:
    print("发现以下问题：")
    for issue in issues:
        print(f"  {issue}")

print("\n建议：")
print("1. 检查位移是否合理（不应过大或过小）")
print("2. 检查目标函数是否单调下降")
print("3. 检查设计变量是否在合理范围内 [0, 1]")
print("4. 检查体积约束是否满足")
print("5. 如果目标函数不下降，检查灵敏度符号是否正确")

