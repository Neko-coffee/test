# AQR参数调优完全指南 🎛️

## 📋 **问题回答**

### **问题1：为什么这两行代码就能裁剪权重？**

```python
# 代码：weight_renderer.py 第171-172行
for view_idx in range(num_views):
    weight_map[:, view_idx] = self._postprocess_weight_map(weight_map[:, view_idx])
```

---

#### **工作原理**

```python
# Step 1: 调用_postprocess_weight_map函数
def _postprocess_weight_map(self, weight_map):
    """
    第379-399行
    """
    # ① 过滤小噪声
    weight_map[weight_map < self.min_weight_threshold] = 0  # 小于0.01的设为0
    
    # ② 🔥 核心：裁剪操作
    if self.normalize_weights:  # 配置中设为True
        weight_map = torch.clamp(weight_map, min=0, max=1.5)  # 🔥 这一行！
        #               ↑ 这个函数会把所有>1.5的值强制改成1.5
    
    return weight_map

# Step 2: clamp函数的作用
torch.clamp(tensor, min=0, max=1.5)
# 等价于：
for each value in tensor:
    if value < 0:
        value = 0
    if value > 1.5:
        value = 1.5  # 🔥 裁剪！
```

---

#### **具体例子**

```python
# 渲染前（多个Query叠加）
weight_map_view0 = [
    [0.0,  0.5,  0.8],
    [1.2,  70.8, 2.3],  # ← 中间这个值是70.8！
    [0.3,  0.9,  0.0]
]

# 调用_postprocess_weight_map后
weight_map_view0 = [
    [0.0,  0.5,  0.8],
    [1.2,  1.5,  1.5],  # ← 70.8被裁剪成1.5，2.3也被裁剪成1.5
    [0.3,  0.9,  0.0]
]

# 为什么之前Camera没有裁剪？
# 因为render_perspective_weights()在第169行直接return了
# 没有调用_postprocess_weight_map！

# 为什么BEV有裁剪？
# 因为render_bev_weights()在第118行调用了_postprocess_weight_map
```

---

### **问题2：下一步参数怎么调整？**

主人的想法：
1. ✅ 调高权重（让AQR作用更明显）
2. ✅ 减少残差连接（降低residual_weight）
3. ❓ 提高裁剪上限（>1.5）

---

#### **方案A：激进调整（推荐尝试）** ⭐⭐⭐⭐

```python
# 配置文件：cmt_aqr_voxel0100_r50_800x320_cbgs.py

# 🔥 1. 调高裁剪上限（更大的权重范围）
renderer_config=dict(
    type='WeightRenderer',
    render_method='gaussian',
    gaussian_sigma=1.0,  # 保持不变
    normalize_weights=True,
    max_weight_clamp=2.5,  # ← 新参数！从1.5提高到2.5
    bev_feature_shape=(128, 128),
    pers_feature_shape=(6, 20, 50)
)

# 🔥 2. 降低残差连接（让AQR作用更大）
modulator_config=dict(
    type='FeatureModulator',
    modulation_type='element_wise',
    normalize_weights=False,
    residual_connection=True,
    residual_weight=0.5,  # ← 从0.7降到0.5（50%原始+50%调制）
    learnable_modulation=False,
    activation='none'
)

# 🔥 3. AQR学习率略微提高
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'pts_backbone': dict(lr_mult=0.1),
            
            # AQR模块
            'aqr_weight_generator': dict(lr_mult=0.8),  # ← 从0.5提高到0.8
            'weight_renderer': dict(lr_mult=0.8),
            'feature_modulator': dict(lr_mult=0.8),
            
            # Transformer及下游
            'transformer': dict(lr_mult=0.8),
            'query_embed': dict(lr_mult=0.8),
            'reference_points': dict(lr_mult=0.5),
            'task_heads': dict(lr_mult=0.8),
            'shared_conv': dict(lr_mult=0.8)
        }
    ),
    weight_decay=0.01
)
```

**预期效果**：
- ✅ AQR调制更明显（residual 70%→50%）
- ✅ 更大的权重动态范围（1.5→2.5）
- ✅ 更快的AQR学习（lr 0.5→0.8）
- ⚠️ 风险：可能不稳定，需要密切监控

---

#### **方案B：保守调整（稳妥选择）** ⭐⭐⭐⭐⭐

```python
# 配置文件：cmt_aqr_voxel0100_r50_800x320_cbgs.py

# 🔥 1. 轻微提高裁剪上限
renderer_config=dict(
    type='WeightRenderer',
    render_method='gaussian',
    gaussian_sigma=1.0,
    normalize_weights=True,
    max_weight_clamp=2.0,  # ← 从1.5提高到2.0（温和提升）
    bev_feature_shape=(128, 128),
    pers_feature_shape=(6, 20, 50)
)

# 🔥 2. 轻微降低残差连接
modulator_config=dict(
    type='FeatureModulator',
    modulation_type='element_wise',
    normalize_weights=False,
    residual_connection=True,
    residual_weight=0.6,  # ← 从0.7降到0.6（60%原始+40%调制）
    learnable_modulation=False,
    activation='none'
)

# 🔥 3. AQR学习率保持当前
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'pts_backbone': dict(lr_mult=0.1),
            
            # AQR模块
            'aqr_weight_generator': dict(lr_mult=0.5),  # ← 保持不变
            'weight_renderer': dict(lr_mult=0.5),
            'feature_modulator': dict(lr_mult=0.5),
            
            # Transformer及下游
            'transformer': dict(lr_mult=0.8),
            'query_embed': dict(lr_mult=0.8),
            'reference_points': dict(lr_mult=0.5),
            'task_heads': dict(lr_mult=0.8),
            'shared_conv': dict(lr_mult=0.8)
        }
    ),
    weight_decay=0.01
)
```

**预期效果**：
- ✅ AQR调制适度增强（residual 70%→60%）
- ✅ 更灵活的权重范围（1.5→2.0）
- ✅ 训练稳定性高
- ✅ 推荐作为下一步尝试

---

#### **方案C：动态调整（高级方案）** ⭐⭐⭐⭐⭐⭐

```python
# 思路：训练过程中动态调整residual_weight

# 配置文件：cmt_aqr_voxel0100_r50_800x320_cbgs.py
modulator_config=dict(
    type='FeatureModulator',
    modulation_type='element_wise',
    normalize_weights=False,
    residual_connection=True,
    residual_weight=0.7,  # ← 初始值
    residual_schedule=dict(
        type='cosine',
        start_value=0.7,  # epoch 0: 70%残差
        end_value=0.4,    # epoch 24: 40%残差
        warmup_epochs=5   # 前5个epoch保持0.7
    ),
    learnable_modulation=False,
    activation='none'
)

# 训练过程：
# Epoch 0-5:   residual=0.7 (70%原始，稳定起步)
# Epoch 6-24:  residual=0.7→0.4 (逐渐增强AQR作用)
```

**优势**：
- ✅ 初期稳定（高残差保护）
- ✅ 后期充分发挥AQR（低残差）
- ✅ 最优的训练策略
- ⚠️ 需要修改代码实现schedule

---

### **问题3：裁剪1.5是否太极端？**

主人的观察：
> Camera权重图每个最大值都在好几十，但训练结果也没有特别糟糕
> 直接全部裁剪成1.5是不是太极端了？

---

#### **历史数据对比**

```python
# 训练很差时（mAP=0.135, NDS=0.419）
# 文件：AQR_TRAINING_ISSUE_ANALYSIS.md 第36-41行

| Iteration | Camera权重图max | 对应性能 |
|-----------|----------------|---------|
| #32025    | 4.04          | mAP很低  |
| #64050    | 8.78 🔥       | mAP极低  |
| #96075    | 1.71          | mAP很低  |
| #128100   | 5.79          | mAP很低  |

# 当前训练（mAP=64.49, NDS=68.49）
| Iteration | Camera权重图max | 对应性能 |
|-----------|----------------|----------|
| 当前      | 70.83 🔥🔥🔥  | mAP中等  |

# 如果裁剪到1.5
| Iteration | Camera权重图max | 预期性能 |
|-----------|----------------|----------|
| 修复后    | 1.50 ✅       | mAP更高？ |
```

---

#### **关键分析**

```python
问题：为什么max=70.8还能有64.49% mAP？

答案1：残差连接的保护作用
  原始特征：mean=0.2, std=0.3
  权重图：mean=0.2, max=70.8
  
  调制特征 = 0.7 × 原始 + 0.3 × (原始 × 权重图)
            = 0.7 × 原始 + 0.3 × 调制部分
            
  即使权重图有极端值70.8，但：
  - 只占很小一部分像素（热点）
  - 大部分区域权重正常（0.2-0.5）
  - 残差连接保留了70%原始特征
  
  → 整体特征分布不会太差

答案2：Transformer的鲁棒性
  - Transformer自带LayerNorm
  - 可以一定程度适应特征分布变化
  - 但适应过程会降低性能

答案3：热点区域的作用
  权重=70.8的区域可能正好是重要区域
  → 虽然数值极端，但方向正确
  → 仍然有一定的检测效果
```

---

#### **为什么还是要裁剪？**

```python
理由1：统计分析
  Camera权重图：
    mean: 0.20
    std: 1.20
    max: 70.83
    
  问题：std=1.20说明分布非常不稳定
       max/mean = 70.83/0.20 = 354倍！
       
  → 极端值会严重扰乱训练梯度

理由2：梯度传播
  特征调制：modulated = original × weight_map
  
  反向传播：
    ∂Loss/∂weight_map = ∂Loss/∂modulated × original
    
  当weight_map=70.8时：
    梯度会被放大70倍！
    → 导致训练不稳定
    → AQR模块难以收敛

理由3：性能上限
  当前max=70.8时：mAP=64.49%
  原模型（无AQR）：mAP=67.9%
  
  → AQR还没超越原模型
  → 说明极端权重阻碍了性能提升
```

---

#### **最优裁剪策略**

```python
# 建议：渐进式裁剪上限

# 配置1：初期训练（Epoch 0-5）
max_weight_clamp=2.5
# 原因：给AQR较大的探索空间，快速找到有效模式

# 配置2：中期训练（Epoch 6-15）
max_weight_clamp=2.0
# 原因：收窄范围，稳定训练，精细调整

# 配置3：后期训练（Epoch 16-24）
max_weight_clamp=1.5
# 原因：严格控制，避免过拟合，提升泛化性能
```

---

## 📊 **完整的参数调优矩阵**

### **场景1：追求最高精度（不在意训练时间）** 🏆

```python
# 训练策略：3阶段
# Stage 1: Epoch 0-8 (探索阶段)
renderer_config=dict(max_weight_clamp=3.0)
modulator_config=dict(residual_weight=0.7)
optimizer_lr_mult=dict(aqr=0.5)

# Stage 2: Epoch 9-16 (收敛阶段)
renderer_config=dict(max_weight_clamp=2.0)
modulator_config=dict(residual_weight=0.5)
optimizer_lr_mult=dict(aqr=0.3)

# Stage 3: Epoch 17-24 (精调阶段)
renderer_config=dict(max_weight_clamp=1.5)
modulator_config=dict(residual_weight=0.4)
optimizer_lr_mult=dict(aqr=0.1)

预期性能：mAP 70-73%, NDS 73-76%
训练时间：24 epochs × 12小时 = 12天
```

---

### **场景2：快速验证（1-2天出结果）** ⚡

```python
# 训练策略：单阶段快速收敛
renderer_config=dict(max_weight_clamp=1.8)  # 温和上限
modulator_config=dict(residual_weight=0.6)  # 平衡残差
optimizer_lr_mult=dict(aqr=0.6)             # 较快学习

# 训练10 epochs即可
预期性能：mAP 66-68%, NDS 70-72%
训练时间：10 epochs × 12小时 = 5天
```

---

### **场景3：在您当前基础上改进** 🎯 **推荐**

```python
# 当前状态：1 epoch，mAP=64.49%, Camera max=70.8
# 目标：修复Bug后继续训练

# Step 1: 修复Bug（已完成）✅
# weight_renderer.py添加了后处理

# Step 2: 调整参数
renderer_config=dict(
    max_weight_clamp=2.0,  # 比1.5宽松，比70.8严格
)

modulator_config=dict(
    residual_weight=0.65,  # 从0.7略微降低
)

optimizer = dict(
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'pts_backbone': dict(lr_mult=0.1),
            'aqr_weight_generator': dict(lr_mult=0.5),
            'transformer': dict(lr_mult=0.8),
            'task_heads': dict(lr_mult=0.8),
        }
    )
)

# Step 3: 继续训练
# 从当前checkpoint继续训练15-20 epochs

预期性能：
  Epoch 5:  mAP 66-67%, NDS 69-70%
  Epoch 10: mAP 68-70%, NDS 71-73%
  Epoch 20: mAP 70-72%, NDS 73-75%
```

---

## 🎛️ **参数速查表**

| 参数 | 当前值 | 保守调整 | 激进调整 | 作用 |
|-----|-------|---------|---------|------|
| **max_weight_clamp** | 1.5 | 2.0 | 2.5-3.0 | 控制权重图上限 |
| **residual_weight** | 0.7 | 0.6 | 0.5-0.4 | 残差连接比例 |
| **gaussian_sigma** | 1.0 | 1.0 | 0.8 | 高斯核宽度 |
| **aqr_lr_mult** | 0.5 | 0.5 | 0.8 | AQR学习率倍数 |
| **transformer_lr_mult** | 0.8 | 0.8 | 1.0 | Transformer学习率 |

---

## ✅ **最终推荐方案**

主人，基于您的训练结果和问题，我推荐：

### **方案：温和改进** ⭐⭐⭐⭐⭐

```python
# 修改配置文件：cmt_aqr_voxel0100_r50_800x320_cbgs.py

# 🔥 1. 提高裁剪上限到2.0（而非1.5）
renderer_config=dict(
    type='WeightRenderer',
    render_method='gaussian',
    gaussian_sigma=1.0,
    normalize_weights=True,
    max_weight_clamp=2.0,  # ← 从1.5改成2.0
    bev_feature_shape=(128, 128),
    pers_feature_shape=(6, 20, 50)
)

# 🔥 2. 略微降低残差（从70%到65%）
modulator_config=dict(
    type='FeatureModulator',
    modulation_type='element_wise',
    normalize_weights=False,
    residual_connection=True,
    residual_weight=0.65,  # ← 从0.7改成0.65
    learnable_modulation=False,
    activation='none'
)

# 🔥 3. 学习率保持不变
# 其他配置不变
```

**理由**：
1. ✅ max=2.0比1.5宽松，不会过度限制AQR
2. ✅ max=2.0比70.8严格，避免梯度爆炸
3. ✅ residual=0.65增强AQR作用（从30%→35%）
4. ✅ 稳妥渐进，风险可控

**预期效果**：
- Camera权重图max: 2.0（从70.8降低）
- Camera相对变化: 15-20%（从33%降低）
- mAP提升: +2-3%（从64.49%→67-68%）
- 训练稳定性: 显著提升

---

**🐾 主人，这是我最推荐的方案！既不极端又能改进性能！✨**

