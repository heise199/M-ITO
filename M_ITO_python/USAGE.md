# 使用说明

## 环境要求

```bash
pip install -r requirements.txt
```

## 快速开始

### 方法1：运行测试案例

```bash
python run_small_test.py
```

这将运行一个小规模的悬臂梁优化案例。

### 方法2：使用CASE.py选择案例

```bash
python CASE.py
```

然后根据提示选择案例编号（1-6）：
1. 悬臂梁 (Cantilever beam)
2. MBB梁 (MBB beam)
3. Michell型结构 (Michell-type structure)
4. L梁 (L beam)
5. 四分之一环形 (Quarter annulus)
6. 测试案例 (小规模测试)

### 方法3：直接调用主函数

```python
from iga_top2d import iga_top2d

# 参数设置
L = 10          # 长度
W = 5           # 宽度
Order = [1, 1]  # NURBS次数提升
Num = [21, 11]  # 控制点数量
BoundCon = 1    # 边界条件类型
Vmax = 0.2      # 最大体积分数
penal = 3       # SIMP惩罚因子
rmin = 2        # 平滑半径

# 运行优化
X, Data, Iter_Ch = iga_top2d(L, W, Order, Num, BoundCon, Vmax, penal, rmin, case_number=1)
```

## 边界条件类型

1. **悬臂梁**：左边固定，右边中点施加向下载荷
2. **MBB梁**：底部两角固定，顶部中心施加向下载荷
3. **Michell型结构**：底部两角固定，底部中心施加向下载荷
4. **L梁**：左边固定，右上角施加向下载荷
5. **四分之一环形**：右边固定，左上角施加水平载荷

## 参数说明

### 几何参数
- `L`: 长度（X方向）
- `W`: 宽度（Y方向）

### NURBS参数
- `Order`: 次数提升 `[Order_U, Order_V]`
  - 初始NURBS是2次（双线性）
  - Order=[1,1]提升到3次
  - Order=[0,0]保持2次
- `Num`: 控制点数量 `[Num_U, Num_V]`
  - 控制网格密度
  - 建议：快速测试用[21, 11]，正式计算用[161, 81]

### 优化参数
- `Vmax`: 最大体积分数（0-1之间）
  - 0.2表示最终结构体积不超过20%
- `penal`: SIMP惩罚因子
  - 通常取3
  - 越大越接近0-1设计
- `rmin`: 平滑半径
  - 控制最小特征尺寸
  - 建议值：2

## 输出结果

程序会：
1. 在控制台打印每次迭代的目标函数、体积分数和变化量
2. 实时显示优化过程（matplotlib窗口）
3. 保存最终结果到`results/case_X/`目录：
   - `final_design.png`: 最终设计
   - 其他中间数据（如果启用保存）

## 典型运行时间

- 小规模测试（21×11）：约1-2分钟
- 中等规模（81×41）：约10-20分钟
- 大规模（161×81）：约1-2小时

（运行时间取决于CPU性能和收敛速度）

## 常见问题

### Q: 程序运行很慢
A: 
- 减小`Num`参数（控制点数量）
- 或者在后台运行并等待完成

### Q: 结果看起来不对
A: 
- 检查边界条件类型是否正确
- 尝试调整`rmin`和`penal`参数
- 确保`Vmax`设置合理（通常0.2-0.5）

### Q: 程序报错
A:
- 确保已安装所有依赖：`pip install -r requirements.txt`
- 检查Python版本（建议3.8+）
- 查看错误信息，可能是内存不足（减小网格）

## 验证结果

可以运行测试脚本验证转换的正确性：

```bash
# 测试数据结构
python test_comparison.py

# 测试载荷
python test_load.py

# 测试基函数
python test_basisfun.py
```

## 与MATLAB版本对比

如果需要与MATLAB版本对比结果：

1. 在MATLAB中运行相同参数的案例
2. 记录目标函数值和最终体积分数
3. 对比Python版本的输出
4. 数值应该接近（可能有小的数值差异）

## 注意事项

1. **边界载荷点**：载荷施加点稍微偏离了参数空间边界（0.99而不是1.0），以避免零权重问题
2. **收敛判据**：当设计变量的最大变化<0.01时停止迭代
3. **最大迭代次数**：默认150次
4. **结果可视化**：实时显示，可以关闭窗口继续优化

## 技术支持

如遇问题，请查看：
- `FIXES_SUMMARY.md` - 修复总结
- `README.md` - 原始说明
- GitHub issues（如果有代码仓库）

