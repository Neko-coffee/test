# CMT DN (去噪训练) 参数位置详解 📍

## 🎯 快速定位

### DN参数定义位置

**文件：** `CMT-master/projects/mmdet3d_plugin/models/dense_heads/cmt_head.py`

**代码位置：** 第 216-220 行（`__init__` 函数参数）

```python
# cmt_head.py - Line 216-220
def __init__(self,
             in_channels,
             num_query=900,
             hidden_dim=128,
             depth_num=64,
             norm_bbox=True,
             downsample_scale=8,
             scalar=10,           # 🔥 DN参数1: 控制groups数量
             noise_scale=1.0,     # 🔥 DN参数2: 噪声尺度
             noise_trans=0.0,     # 🔥 DN参数3: 平移噪声
             dn_weight=1.0,       # 🔥 DN参数4: DN损失权重
             split=0.75,          # 🔥 DN参数5: 掩码阈值
             ...):
```

---

## 📋 DN参数详细说明

### 1. `scalar` (默认=10)

**作用：** 控制DN query的重复次数（groups数量）

**代码位置：** Line 275, 493

```python
# 初始化
self.scalar = scalar  # Line 275

# 使用
groups = min(self.scalar, self.num_query // max(known_num))  # Line 493
```

**计算逻辑：**
```
groups = min(scalar, num_query // 每个样本的GT数量)

例如：
  - num_query = 900
  - 某个样本有30个GT
  - groups = min(10, 900 // 30) = min(10, 30) = 10
  
  → 每个GT会被复制10次，加噪声后作为DN query
  → 总DN query = 30 × 10 = 300个
  → 总query = 900原始 + 300 DN = 1200个
```

**影响：**
- 值越大 → DN query越多 → 训练更强调去噪能力
- 值越小 → DN query越少 → 训练更接近普通DETR

---

### 2. `noise_scale` (默认=1.0)

**作用：** 控制添加到GT bbox中心点的噪声强度

**代码位置：** Line 276, 502-506

```python
# 初始化
self.bbox_noise_scale = noise_scale  # Line 276

# 使用 (Line 502-506)
if self.bbox_noise_scale > 0:
    diff = known_bbox_scale / 2 + self.bbox_noise_trans
    rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0  # [-1, 1]
    known_bbox_center += torch.mul(rand_prob, diff) * self.bbox_noise_scale
```

**计算逻辑：**
```
噪声范围 = (bbox_scale / 2 + bbox_noise_trans) × noise_scale

例如（汽车）：
  - bbox_scale = [4m, 2m, 1.5m]  (长宽高)
  - noise_scale = 1.0
  - bbox_noise_trans = 0.0
  
  噪声范围 = [2m, 1m, 0.75m]
  
  → 中心点在 ±2m (x), ±1m (y), ±0.75m (z) 范围内随机偏移
```

**影响：**
- 值越大 → 噪声越强 → 模型需要更强的去噪能力
- 值越小 → 噪声越弱 → 降低难度

---

### 3. `noise_trans` (默认=0.0)

**作用：** 额外的平移噪声偏置

**代码位置：** Line 277, 503

```python
# 初始化
self.bbox_noise_trans = noise_trans  # Line 277

# 使用 (Line 503)
diff = known_bbox_scale / 2 + self.bbox_noise_trans
```

**计算逻辑：**
```
噪声范围 = (bbox_scale / 2 + noise_trans) × noise_scale

增加noise_trans可以扩大噪声范围，即使对小目标也能添加足够的噪声
```

**影响：**
- 通常保持0.0
- 可以设置小值（如0.5）来增加小目标的噪声

---

### 4. `dn_weight` (默认=1.0)

**作用：** DN损失的权重系数

**代码位置：** Line 278, 958

```python
# 初始化
self.dn_weight = dn_weight  # Line 278

# 使用 (Line 958)
return self.dn_weight * loss_cls, self.dn_weight * loss_bbox
```

**计算逻辑：**
```
最终DN损失 = dn_weight × (DN分类损失 + DN回归损失)

总损失 = 普通query损失 + DN损失
```

**影响：**
- 值越大 → DN损失权重越高 → 模型更关注去噪任务
- 值越小 → DN损失权重越低 → 降低DN影响

---

### 5. `split` (默认=0.75)

**作用：** 控制哪些DN query被标记为"difficult"（困难样本）

**代码位置：** Line 279, 511-512, 930

```python
# 初始化
self.split = split  # Line 279

# 使用 (Line 511-512)
mask = torch.norm(rand_prob, 2, 1) > self.split  # rand_prob在[-1,1]之间
known_labels[mask] = sum(self.num_classes)       # 标记为背景类

# 平均因子计算 (Line 930)
cls_avg_factor = num_tgt * 3.14159 / 6 * split * split * split
```

**计算逻辑：**
```
对于每个DN query:
  1. 生成随机噪声向量 rand_prob ∈ [-1, 1]³
  2. 计算L2范数: norm = sqrt(x² + y² + z²)
  3. 如果 norm > split（默认0.75）:
     → 标记为"困难样本"（背景类）
     → 模型需要识别这是噪声过大的伪目标
  
概率计算：
  - split = 0.75 时，约 15-20% 的DN query被标记为困难样本
  - split = 0.5  时，约 50% 的DN query被标记为困难样本
```

**影响：**
- 值越大（接近1.0）→ 困难样本越少 → 降低难度
- 值越小（接近0.0）→ 困难样本越多 → 增加难度

---

## 🔍 DN参数在哪里配置？

### ❌ 不在配置文件中！

```python
# 检查所有配置文件
CMT-master/projects/configs/fusion/*.py
  ❌ 没有找到 noise_scale
  ❌ 没有找到 dn_weight
  ❌ 没有找到 scalar
  ❌ 没有找到 split
```

### ✅ 使用默认值！

**原因：** CMT的配置文件使用继承和默认值机制

```python
# 配置文件中只写：
model = dict(
    pts_bbox_head=dict(
        type='CmtHead',
        # 其他参数...
    )
)

# CmtHead的__init__中已经有默认值：
def __init__(self, ..., scalar=10, noise_scale=1.0, ...):
    pass

→ 如果配置文件不指定，就使用默认值
```

---

## 📝 如何修改DN参数？

### 方法1: 在配置文件中显式指定

```python
# cmt_aqr_voxel0100_r50_800x320_cbgs.py

model = dict(
    pts_bbox_head=dict(
        type='CmtHead',
        
        # === DN训练参数 ===
        scalar=10,           # groups倍数
        noise_scale=1.0,     # 噪声强度
        noise_trans=0.0,     # 平移噪声
        dn_weight=1.0,       # DN损失权重
        split=0.75,          # 困难样本阈值
        
        # === 其他参数 ===
        num_query=900,
        # ...
    )
)
```

### 方法2: 禁用DN训练

```python
# 如果想禁用DN，设置noise_scale=0
model = dict(
    pts_bbox_head=dict(
        type='CmtHead',
        noise_scale=0.0,     # 🔥 禁用DN训练
        # 其他参数保持默认
    )
)
```

---

## 🧪 DN参数对Query数量的影响

### 计算公式

```python
# 在prepare_for_dn函数中 (Line 493-516)

groups = min(scalar, num_query // max(known_num))
pad_size = max(known_num) × groups

总Query数 = num_query (原始) + pad_size (DN)
```

### 实际例子

**场景：** batch_size=4, 每个样本的GT数量为 [25, 30, 28, 32]

```python
# 计算过程
max(known_num) = 32  # 最多的GT数量
groups = min(10, 900 // 32) = min(10, 28) = 10
pad_size = 32 × 10 = 320

总Query数 = 900 + 320 = 1220

# 但是！从日志看到是1730，说明：
实际max(known_num) ≈ 83
groups = min(10, 900 // 83) = min(10, 10) = 10
pad_size = 83 × 10 = 830
总Query数 = 900 + 830 = 1730 ✅ 吻合！
```

### 不同参数下的Query数量

| scalar | max_GT | groups | pad_size | 总Query | 说明 |
|--------|--------|--------|----------|---------|------|
| 10 | 30 | 10 | 300 | 1200 | 标准配置 |
| 10 | 83 | 10 | 830 | 1730 | 实际观测 |
| 5 | 83 | 5 | 415 | 1315 | 减少DN |
| 0 | - | 0 | 0 | 900 | 禁用DN |

---

## 🆚 CMT vs MoME 的DN对比

```
┌──────────────────┬─────────────┬──────────────┐
│ 项目             │ CMT         │ MoME         │
├──────────────────┼─────────────┼──────────────┤
│ 使用DN训练？     │ ✅ Yes      │ ❌ No        │
│ scalar           │ 10          │ -            │
│ noise_scale      │ 1.0         │ -            │
│ dn_weight        │ 1.0         │ -            │
│ split            │ 0.75        │ -            │
│ 实际Query数      │ 1730 (动态) │ 固定         │
│ 训练复杂度       │ 高          │ 中           │
└──────────────────┴─────────────┴──────────────┘

结论：
  - CMT使用DN训练，加速收敛和提高鲁棒性
  - MoME不使用DN，但多专家机制本身就很强
  - AQR在CMT上失败可能与DN的复杂性有关
```

---

## 💡 对AQR的启示

### 问题：AQR权重生成时需要考虑DN query吗？

**当前实现：** 
```python
# aqr_weight_generator.py
# AQR对所有query生成权重，包括DN query

lidar_weights, camera_weights = self.aqr_weight_generator(
    query_embed,  # shape: [1730, bs, C]  包含900原始 + 830 DN
    memory, 
    ...
)
```

**潜在问题：**
```
1. DN query是噪声GT，不是真实检测query
   → 为噪声query生成权重有意义吗？
   
2. DN query的权重可能干扰权重图渲染
   → 830个噪声query的权重被渲染到特征图上
   → 可能引入额外的噪声
   
3. 训练/推理不一致
   → 训练时：1730个query（含DN）
   → 推理时：900个query（无DN）
   → AQR在两个阶段看到的query数量不同
```

**可能的改进：**
```python
# 方案A：只为原始query生成权重
if self.training and mask_dict is not None:
    pad_size = mask_dict['pad_size']
    # 只取后900个原始query
    original_query = query_embed[pad_size:]
    weights = self.aqr_weight_generator(original_query, ...)
else:
    weights = self.aqr_weight_generator(query_embed, ...)

# 方案B：禁用DN训练（用于AQR实验）
model = dict(
    pts_bbox_head=dict(
        noise_scale=0.0,  # 禁用DN
        # AQR配置...
    )
)
```

---

**🐾 总结：DN参数在 `cmt_head.py` 的 `__init__` 中定义，使用默认值。如果要修改，需要在配置文件中显式指定！**

