# IGA拓扑优化 - Python版本

这是MATLAB版本的IGA拓扑优化代码的Python实现。

## 依赖库

- numpy
- scipy
- matplotlib
- geomdl (NURBS库)

## 安装

```bash
conda activate nurbs
pip install -r requirements.txt
```

## 使用方法

运行主程序：

```bash
python CASE.py
```

或者直接运行单个案例：

```python
from iga_top2d import iga_top2d

# Cantilever beam
iga_top2d(10, 5, [1, 1], [161, 81], 1, 0.2, 3, 2)
```

## 参数说明

- L: 长度
- W: 宽度
- Order: B样条阶数 [p, q]
- Num: 控制点数量 [n, m]
- BoundCon: 边界条件类型 (1-5)
  - 1: Cantilever beam
  - 2: MBB beam
  - 3: Michell-type structure
  - 4: L beam
  - 5: Quarter annulus
- Vmax: 最大体积分数
- penal: 惩罚参数
- rmin: 最小过滤半径

## 文件结构

- `CASE.py`: 主入口文件，包含多个测试案例
- `iga_top2d.py`: 主程序，IGA拓扑优化核心算法
- `geom_mod.py`: 几何建模模块
- `pre_iga.py`: IGA预处理模块
- `boun_cond.py`: 边界条件模块
- `shep_fun.py`: 形状函数（密度过滤）模块
- `guadrature.py`: 高斯积分模块
- `nrbbasisfun.py`: NURBS基函数计算模块
- `nrbbasisfunder.py`: NURBS基函数导数计算模块
- `stiff_ele2d.py`: 单元刚度矩阵模块
- `stiff_ass2d.py`: 刚度矩阵组装模块
- `solving.py`: 线性方程组求解模块
- `oc.py`: 优化准则模块
- `plot_data.py`: 绘图数据准备模块
- `plot_topy.py`: 拓扑优化结果绘图模块

## 注意事项

1. 确保已激活conda环境 `nurbs`
2. 确保已安装 `geomdl` 库
3. 代码中的索引转换：MATLAB使用1-based索引，Python使用0-based索引
4. 某些geomdl API可能需要根据实际版本进行调整

