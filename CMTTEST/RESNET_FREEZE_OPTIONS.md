# ResNet50 冻结选项说明 🔧

**骨干网络**: ResNet50  
**结构**: 4个stage（stage 0, 1, 2, 3）

---

## 📊 **ResNet50 结构**

```
ResNet50:
├── Conv1 + BN + ReLU + MaxPool (stem)
├── Stage 0 (Layer1): 3 blocks  - 256 channels
├── Stage 1 (Layer2): 4 blocks  - 512 channels
├── Stage 2 (Layer3): 6 blocks  - 1024 channels
└── Stage 3 (Layer4): 3 blocks  - 2048 channels
```

---

## ⚙️ **冻结选项**

### **选项1：完全冻结（推荐）** ⭐⭐⭐⭐⭐

```python
img_backbone=dict(
    frozen_stages=3,    # 冻结stage 0,1,2,3（全部）
    norm_eval=True,
)

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),  # 完全不更新
        }
    )
)
```

**效果**：
- ✅ 所有ResNet参数冻结
- ✅ 训练速度最快
- ✅ 显存占用最低
- ✅ 特征最稳定

### **选项2：部分冻结（当前配置）** ⭐⭐⭐⭐

```python
img_backbone=dict(
    frozen_stages=2,    # 冻结stage 0,1,2（保留stage3）
    norm_eval=True,
)

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),  # 仍然不更新（通过lr_mult控制）
        }
    )
)
```

**效果**：
- ✅ Stage 0,1,2 冻结（基础特征保留）
- ⚠️ Stage 3 可以微调（高层语义特征）
- ⚠️ 但由于`lr_mult=0.0`，实际上stage3也不会更新

### **选项3：激进微调** ⭐⭐

```python
img_backbone=dict(
    frozen_stages=1,    # 只冻结stage 0,1
    norm_eval=True,
)

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.01),  # 1%学习率微调stage 2,3
        }
    )
)
```

**效果**：
- ⚠️ Stage 2,3 微调（可能破坏预训练特征）
- ⚠️ 训练不稳定风险

---

## 🎯 **推荐配置（完全冻结）**

```python
# 配置文件
model = dict(
    img_backbone=dict(
        frozen_stages=3,    # 🔥 冻结所有stage
        norm_eval=True,     # 🔥 BN保持eval
    ),
    pts_backbone=dict(
        frozen_stages=2,    # 🔥 SECOND冻结前2层（共3层）
    ),
)

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),      # 🔥 完全冻结
            'pts_backbone': dict(lr_mult=0.0),      # 🔥 完全冻结
            'pts_voxel_encoder': dict(lr_mult=0.0),
            'pts_middle_encoder': dict(lr_mult=0.0),
        }
    )
)
```

---

## 📋 **frozen_stages 参数说明**

| frozen_stages | 冻结的stage | 可训练的stage | 说明 |
|--------------|------------|--------------|------|
| **-1** | 无 | 0,1,2,3 | 全部可训练 |
| **0** | stem | 0,1,2,3 | 只冻结stem |
| **1** | stem, stage0 | 1,2,3 | 冻结低层特征 |
| **2** | stem, stage0,1 | 2,3 | 冻结中低层特征 |
| **3** | stem, stage0,1,2 | 3 | 冻结大部分 |
| **4** | stem, stage0,1,2,3 | 无 | 全部冻结 |

**注意**：ResNet的`frozen_stages`参数范围是`-1`到`4`。

---

## 🔍 **如何选择？**

### **场景1：使用CMT预训练权重（当前情况）**
```python
# 推荐：完全冻结
frozen_stages=3  # 或 4（取决于实现）
lr_mult=0.0
```
**原因**：CMT已经学到了高质量特征，不需要再调整

### **场景2：数据分布差异大**
```python
# 备选：部分冻结
frozen_stages=2
lr_mult=0.01  # stage3极低学习率微调
```
**原因**：允许高层特征适应新数据

### **场景3：从头训练（不适用）**
```python
frozen_stages=-1
lr_mult=1.0
```
**原因**：需要学习所有特征

---

## ⚠️ **当前配置分析**

**您当前的配置**：
```python
img_backbone=dict(
    frozen_stages=3,    # 冻结stage 0,1,2，stage3可训练
    norm_eval=True,
)

optimizer = dict(
    'img_backbone': dict(lr_mult=0.0),  # 但学习率是0
)
```

**实际效果**：
- `frozen_stages=3` → stage3的`requires_grad=True`（可训练）
- `lr_mult=0.0` → stage3的学习率=0（不更新）

**结果**：
- ✅ 参数不会更新（正确）
- ⚠️ 但仍然计算梯度（浪费计算）

**建议修改**：
```python
img_backbone=dict(
    frozen_stages=4,    # 🔥 改为4，完全冻结（包括stage3）
    norm_eval=True,
)
```

---

## 🚀 **最终推荐**

```python
# 🔥 完全冻结所有stage（最优）
model = dict(
    img_backbone=dict(
        frozen_stages=4,    # ResNet50全部冻结
        norm_eval=True,
    ),
    pts_backbone=dict(
        frozen_stages=3,    # SECOND全部冻结（如果支持）
    ),
)

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'pts_backbone': dict(lr_mult=0.0),
            'pts_voxel_encoder': dict(lr_mult=0.0),
            'pts_middle_encoder': dict(lr_mult=0.0),
        }
    )
)
```

**效果**：
1. ✅ `frozen_stages` → `requires_grad=False`（不计算梯度）
2. ✅ `lr_mult=0.0` → 学习率=0（双重保险）
3. ✅ 最快的训练速度
4. ✅ 最低的显存占用

---

**主人，建议将`frozen_stages=3`改为`frozen_stages=4`以完全冻结ResNet！** 🎯



