# IgaTop2D - 基于等几何分析的2D拓扑优化程序

这是从 MATLAB 转换为 Python 的等几何拓扑优化程序。该程序实现了基于 NURBS（Non-Uniform Rational B-Splines）的结构拓扑优化。

## 功能特性

- ✅ 完整的 NURBS 工具箱实现（从 MATLAB NURBS Toolbox 转换）
- ✅ 等几何分析（IGA）框架
- ✅ SIMP（Solid Isotropic Material with Penalization）材料插值
- ✅ 优化准则（OC）方法
- ✅ Shepard 函数平滑机制
- ✅ 实时可视化（matplotlib）
- ✅ 支持多种经典拓扑优化案例

## 快速开始

### 1. 安装依赖

```bash
cd M_ITO_python
pip install -r requirements.txt
```

需要的依赖包：
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0

### 2. 测试基础功能

```bash
python test_basic.py
```

### 3. 运行优化案例

```bash
python CASE.py
```

程序会提示选择案例（1-6）。建议先运行案例 6（测试案例）快速验证。

## 支持的案例

1. **悬臂梁 (Cantilever beam)** - 经典悬臂梁拓扑优化
2. **MBB 梁 (MBB beam)** - 对称 MBB 梁结构
3. **Michell 型结构** - Michell 最优桁架结构
4. **L 梁** - L 形结构优化
5. **四分之一环形** - 圆形域拓扑优化
6. **测试案例** - 小规模快速测试

## 项目结构

```
M_ITO_python/
├── nurbs/              # NURBS 工具箱
│   ├── bspline.py     # B-spline 基础函数
│   ├── nrb_core.py    # NURBS 核心功能
│   └── nrb_ops.py     # NURBS 操作（节点插入、次数提升）
├── CASE.py            # 主运行脚本
├── iga_top2d.py       # 主优化函数
├── test_basic.py      # 基础功能测试
├── requirements.txt   # 依赖包列表
├── 使用说明.md         # 详细使用说明
└── [其他模块文件]
```

详细使用说明请参阅 [使用说明.md](./使用说明.md)

## 输出示例

程序运行时会显示：
```
 It.:    1 Obj.:   245.8932 Vol.:  1.000 ch.:  0.200
 It.:    2 Obj.:   198.4521 Vol.:  0.500 ch.:  0.150
 ...
```

同时会显示 5 个实时更新的图形窗口：
- 控制点密度 3D 图
- 高斯点密度 3D 图
- 密度分布函数曲面
- 材料分布图
- 最终拓扑结构

## 与原 MATLAB 版本的对应关系

| MATLAB | Python |
|--------|--------|
| IgaTop2D.m | iga_top2d.py |
| CASE.m | CASE.py |
| Pre_IGA.m | pre_iga.py |
| Boun_Cond.m | boun_cond.py |
| Geom_Mod.m | geom_mod.py |
| NURBS Toolbox | nurbs/ 模块 |

## 主要技术特点

1. **1:1 复刻**: 严格按照 MATLAB 代码逻辑转换，保持算法一致性
2. **索引转换**: 正确处理 MATLAB 1-based 到 Python 0-based 的索引转换
3. **稀疏矩阵**: 使用 scipy.sparse 高效处理大规模稀疏矩阵
4. **向量化**: 充分利用 NumPy 向量化操作提高性能
5. **调试信息**: 保留详细的调试输出，便于验证和排错

## 注意事项

- 大规模案例（如案例 1、2）可能需要较长计算时间和较大内存
- 建议先运行案例 6 测试程序是否正常工作
- 程序包含详细的调试输出，有助于理解执行过程

## 主要参考文献

(1) Jie Gao, Lin Wang, Zhen Luo, Liang Gao. IgaTop: an implementation of topology optimization for structures using IGA in Matlab. Structural and Multidisciplinary Optimization.

(2) Jie Gao, Liang Gao, Zhen Luo, Peigen Li. Isogeometric topology optimization for continuum structures using density distribution function. Int J Numer Methods Eng, 2019, 119:991–1017

## 开发信息

- **原始 MATLAB 代码**: Jie Gao (JieGao@hust.edu.cn)
- **Python 转换**: 2025
- **转换方式**: 1:1 逐行复刻，保持算法完整性

## 许可证

本程序仅供学术和教育用途。作者不保证代码没有错误，并且不对因使用程序而引起的任何事件承担责任。

