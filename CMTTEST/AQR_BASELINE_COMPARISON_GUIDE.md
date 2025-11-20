# AQR vs 基线模型对比实验指南 🔬

## 🎯 实验目的

通过对比**使用AQR**和**不使用AQR**的模型，量化AQR权重图渲染机制对性能的实际影响。

---

## 📁 配置文件说明

### 1. 基线模型（不使用AQR）
```bash
配置文件：projects/configs/fusion/cmt_baseline_voxel0100_r50_800x320_cbgs.py

关键配置：
  enable_aqr=False  ❌ 关闭AQR
  
工作目录：
  work_dirs/cmt_baseline_voxel0100_r50_800x320_cbgs/
```

### 2. AQR模型（使用AQR）
```bash
配置文件：projects/configs/fusion/cmt_aqr_voxel0100_r50_800x320_cbgs.py

关键配置：
  enable_aqr=True   ✅ 启用AQR
  max_weight_clamp=2.5
  residual_weight=0.6
  
工作目录：
  work_dirs/cmt_aqr_voxel0100_r50_800x320_cbgs/
```

---

## 🚀 实验步骤

### Step 1: 训练基线模型

```bash
# 方法1：分布式训练（推荐，8卡）
bash tools/dist_train.sh \
    projects/configs/fusion/cmt_baseline_voxel0100_r50_800x320_cbgs.py \
    8 \
    --work-dir work_dirs/cmt_baseline_voxel0100_r50_800x320_cbgs

# 方法2：单卡训练
python tools/train.py \
    projects/configs/fusion/cmt_baseline_voxel0100_r50_800x320_cbgs.py \
    --work-dir work_dirs/cmt_baseline_voxel0100_r50_800x320_cbgs
```

**预期训练时间：** ~2.5小时（1 epoch，8卡）

### Step 2: 训练AQR模型

```bash
# 分布式训练（8卡）
bash tools/dist_train.sh \
    projects/configs/fusion/cmt_aqr_voxel0100_r50_800x320_cbgs.py \
    8 \
    --work-dir work_dirs/cmt_aqr_voxel0100_r50_800x320_cbgs
```

**预期训练时间：** ~2.5小时（1 epoch，8卡）

### Step 3: 性能评估

```bash
# 评估基线模型
python tools/test.py \
    projects/configs/fusion/cmt_baseline_voxel0100_r50_800x320_cbgs.py \
    work_dirs/cmt_baseline_voxel0100_r50_800x320_cbgs/latest.pth \
    --eval bbox

# 评估AQR模型
python tools/test.py \
    projects/configs/fusion/cmt_aqr_voxel0100_r50_800x320_cbgs.py \
    work_dirs/cmt_aqr_voxel0100_r50_800x320_cbgs/latest.pth \
    --eval bbox
```

---

## 📊 对比维度

### 1. 主要性能指标

| 指标 | 基线模型 | AQR模型 | 差异 | 评价 |
|-----|---------|---------|------|-----|
| **mAP** | ? | 63.97% | ? | 主要指标 |
| **NDS** | ? | 68.25% | ? | 主要指标 |
| mATE | ? | 0.3373 | ? | 定位误差 |
| mASE | ? | 0.2500 | ? | 尺度误差 |
| mAOE | ? | 0.3259 | ? | 方向误差 |
| mAVE | ? | 0.2731 | ? | 速度误差 |
| mAAE | ? | 0.1876 | ? | 属性误差 |

### 2. 各类别AP对比

| 类别 | 基线AP | AQR AP | 差异 | 分析 |
|-----|--------|--------|------|-----|
| car | ? | 0.851 | ? | 最常见类别 |
| truck | ? | 0.602 | ? | 大型车辆 |
| bus | ? | 0.729 | ? | 长型车辆 |
| trailer | ? | 0.416 | ? | 困难类别 |
| construction_vehicle | ? | 0.289 | ? | 最困难类别 |
| pedestrian | ? | 0.822 | ? | 小目标 |
| motorcycle | ? | 0.695 | ? | 小型车辆 |
| bicycle | ? | 0.594 | ? | 小目标 |
| traffic_cone | ? | 0.692 | ? | 小目标 |
| barrier | ? | 0.708 | ? | 静态障碍物 |

### 3. 训练效率对比

| 维度 | 基线模型 | AQR模型 | 差异 |
|-----|---------|---------|------|
| 训练时间/epoch | ? | ~2.5h | ? |
| 内存占用 | ? | ~22GB | ? |
| 推理速度 | ? | ? | ? |
| 参数量 | ? | ? | ? |

### 4. 训练稳定性对比

| 维度 | 基线模型 | AQR模型 | 评价 |
|-----|---------|---------|-----|
| 损失收敛 | ? | 13.35→10.12 | ✅ 平滑 |
| 梯度稳定性 | ? | grad_norm~45 | ✅ 稳定 |
| NaN/Inf出现 | ? | 无 | ✅ 稳定 |

---

## 🔍 深度分析维度

### 1. AQR权重分布分析（仅AQR模型）

```python
# 从训练日志提取
LiDAR权重均值：0.734 (73.4%)
Camera权重均值：0.821 (82.1%)

分析：
  - Camera权重略高于LiDAR
  - 说明模型更倾向使用Camera特征
  - 符合预期（Camera提供丰富语义信息）
```

### 2. 特征调制效果分析（仅AQR模型）

```python
BEV特征调制：
  relative_change: 10.58%
  
Perspective特征调制：
  relative_change: 36.09%

分析：
  - Camera调制强度远高于LiDAR
  - 可能是性能瓶颈所在
```

### 3. 各距离段性能对比

| 距离段 | 基线AP | AQR AP | 差异 | 分析 |
|-------|--------|--------|------|-----|
| 0-20m | ? | ? | ? | 近距离 |
| 20-40m | ? | ? | ? | 中距离 |
| 40-60m | ? | ? | ? | 远距离 |

---

## 📈 结果分析模板

### 场景1：AQR显著提升性能

```
假设结果：
  基线mAP: 61.5%
  AQR mAP: 63.97%
  提升: +2.47%

结论：
  ✅ AQR权重图渲染有效
  ✅ 多模态融合得到优化
  ✅ 值得继续优化AQR参数

下一步：
  - 尝试更优的max_weight_clamp和residual_weight组合
  - 分析哪些类别受益最多
  - 研究AQR在不同场景下的表现
```

### 场景2：AQR略微提升性能

```
假设结果：
  基线mAP: 63.0%
  AQR mAP: 63.97%
  提升: +0.97%

结论：
  ⚠️ AQR有轻微提升但不显著
  ⚠️ 需要权衡计算开销
  ⚠️ 可能需要调整参数

下一步：
  - 尝试不同的window_sizes
  - 调整max_weight_clamp和residual_weight
  - 分析计算开销是否值得
```

### 场景3：AQR未带来提升或性能下降

```
假设结果：
  基线mAP: 65.5%
  AQR mAP: 63.97%
  下降: -1.53%

结论：
  ❌ 当前AQR配置不适合
  ❌ 可能过度调制特征
  ❌ 需要重新审视设计

下一步：
  - 回到更保守的配置（2.0+0.70）
  - 检查是否有实现bug
  - 考虑简化AQR设计
```

---

## 🎯 关键对比点

### 1. 性能提升是否显著？

**判断标准：**
- ✅ 显著提升：mAP提升 > 2%，NDS提升 > 1.5%
- ⚠️ 轻微提升：mAP提升 0.5-2%，NDS提升 0.5-1.5%
- ❌ 无提升/下降：mAP提升 < 0.5%或为负

### 2. 哪些类别受益最多？

**重点关注：**
- 小目标：pedestrian, bicycle, traffic_cone
- 困难类别：trailer, construction_vehicle
- 常见类别：car, truck

### 3. 计算开销是否可接受？

**评估维度：**
- 训练时间增加：< 10%可接受
- 内存增加：< 15%可接受
- 推理速度下降：< 5%可接受

### 4. 训练稳定性如何？

**评估维度：**
- 损失收敛速度
- 梯度稳定性
- 是否出现NaN/Inf

---

## 📝 实验记录模板

```markdown
## 实验日期：2025-10-12

### 实验配置
- 基线模型：cmt_baseline_voxel0100_r50_800x320_cbgs
- AQR模型：cmt_aqr_voxel0100_r50_800x320_cbgs (2.5+0.6)
- 训练轮数：1 epoch
- GPU数量：8卡

### 性能对比
| 指标 | 基线 | AQR | 差异 |
|-----|------|-----|------|
| mAP | XX.XX% | 63.97% | +/-X.XX% |
| NDS | XX.XX% | 68.25% | +/-X.XX% |

### 各类别AP对比
（填写表格）

### 结论
（填写分析）

### 下一步计划
（填写计划）
```

---

## 🔧 常见问题排查

### Q1: 基线模型训练失败

**可能原因：**
- 配置文件路径错误
- 预训练权重未加载
- GPU内存不足

**解决方案：**
```bash
# 检查配置文件
cat projects/configs/fusion/cmt_baseline_voxel0100_r50_800x320_cbgs.py

# 检查GPU状态
nvidia-smi

# 减少batch size（如果内存不足）
# 在配置文件中修改：
data = dict(samples_per_gpu=1)  # 改为1
```

### Q2: 两个模型性能相近

**可能原因：**
- AQR参数设置不当
- 训练轮数不够
- 数据集太小

**解决方案：**
- 尝试不同的AQR参数组合
- 增加训练轮数到2-3 epochs
- 检查AQR是否真正生效（查看debug日志）

### Q3: AQR模型性能反而下降

**可能原因：**
- 过度调制（residual_weight太低）
- 裁剪上限太高（max_weight_clamp太大）
- 实现有bug

**解决方案：**
- 回到保守配置（2.0+0.70）
- 检查debug日志中的调制强度
- 验证权重图渲染是否正确

---

## 🎓 实验总结建议

完成对比实验后，建议从以下角度总结：

1. **性能提升量化**
   - 主要指标（mAP, NDS）的绝对提升
   - 各类别AP的变化趋势
   - 统计显著性分析

2. **计算开销评估**
   - 训练时间对比
   - 推理速度对比
   - 内存占用对比

3. **适用场景分析**
   - AQR在哪些场景下有效
   - 哪些类别受益最多
   - 是否值得部署到生产环境

4. **改进方向**
   - 参数调优建议
   - 架构改进思路
   - 未来研究方向

---

**🐾 主人，这份指南将帮助您系统地对比AQR和基线模型，量化AQR的实际效果！祝实验顺利！**

