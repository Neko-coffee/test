# AQR 不同分辨率配置对比说明 📐

## 🎯 两种分辨率配置概览

| 配置项 | 800×320 (ResNet50) | 1600×640 (VoVNet) | 说明 |
|-------|-------------------|-------------------|------|
| **配置文件** | `cmt_aqr_voxel0100_r50_800x320_cbgs.py` | `cmt_aqr_voxel0075_vov_1600x640_cbgs.py` | - |
| **图像尺寸** | 800×320 | 1600×640 | 1600x640是标准分辨率 |
| **体素大小** | 0.1m | 0.075m | 0.075m更精细 |
| **图像骨干** | ResNet50 | VoVNet-99 | VoVNet更强大 |

---

## 🔧 关键参数对比

### 1. **特征图尺寸** 📊

#### **透视特征图（Camera）**
```python
# 800×320配置
pers_feature_shape=(6, 20, 50)   # 800/16=50, 320/16=20

# 1600×640配置
pers_feature_shape=(6, 40, 100)  # 1600/16=100, 640/16=40
```

**计算公式**：`特征图尺寸 = 图像尺寸 / 16`（下采样16倍）

---

#### **BEV特征图（LiDAR）**
```python
# 800×320配置（voxel_size=0.1）
bev_feature_shape=(128, 128)     # 1024/8=128（点云范围108m，体素0.1m，grid=1080，下采样8倍）

# 1600×640配置（voxel_size=0.075）
bev_feature_shape=(180, 180)     # 1440/8=180（点云范围108m，体素0.075m，grid=1440，下采样8倍）
```

**计算公式**：
```
1. grid_size = point_cloud_range / voxel_size = 108 / 0.1 = 1080（或108/0.075=1440）
2. bev_feature_size = grid_size / out_size_factor = 1080 / 8 = 135（但实际使用128或180）
```

---

### 2. **局部窗口大小** 🪟

```python
# 800×320配置
window_sizes=[8, 5]      # [camera_window=8, lidar_window=5]
window_size=8            # attention_bias_generator中使用

# 1600×640配置
window_sizes=[15, 5]     # [camera_window=15, lidar_window=5]（camera窗口约翻倍）
window_size=15           # attention_bias_generator中使用
```

**设计原理**：
- **Camera窗口**：与特征图分辨率正相关
  - 800×320 → 特征图50×20 → window=8
  - 1600×640 → 特征图100×40 → window=15（约1.875倍）
  
- **LiDAR窗口**：保持不变（BEV特征图变化不大）
  - 128×128 → window=5
  - 180×180 → window=5（仍然合理）

---

### 3. **高斯核参数** 🔵

```python
# 800×320配置
gaussian_sigma=1.0       # 小特征图用小sigma

# 1600×640配置
gaussian_sigma=2.0       # 大特征图用大sigma（翻倍）
bilinear_radius=2.0      # 双线性插值半径也增大
```

**原因**：
- 特征图越大，每个像素的感受野相对越小
- 需要增大sigma以覆盖合理的空间范围
- 保持相对比例：`sigma / feature_size` 大致恒定

---

### 4. **学习率调整** 📈

```python
# 800×320配置
lr=0.00014               # 较低分辨率用较低学习率

# 1600×640配置
lr=0.0002                # 较高分辨率用稍高学习率（约1.4倍）
```

**原因**：
- 更高分辨率 → 更多特征 → 梯度更充分
- 可以承受稍高的学习率以加速收敛
- 但不宜过高，因为模型复杂度也增加了

---

### 5. **批次大小** 📦

```python
# 800×320配置
samples_per_gpu=1        # 已经是最小值

# 1600×640配置
samples_per_gpu=1        # 内存占用大，保持batch_size=1
```

**内存占用估算**：
- 800×320：特征图 6×20×50×256 ≈ 1.5M参数
- 1600×640：特征图 6×40×100×256 ≈ 6.1M参数（约4倍）

---

## 🎯 配置选择建议

### **使用800×320配置的场景**：
- ✅ GPU内存有限（<16GB）
- ✅ 快速实验和验证
- ✅ 需要更快的训练速度
- ✅ 数据集较小

### **使用1600×640配置的场景**：
- ✅ 追求最高精度
- ✅ GPU内存充足（≥24GB）
- ✅ 正式训练和评测
- ✅ 需要检测小目标

---

## 📊 性能对比（预期）

| 指标 | 800×320 | 1600×640 | 增益 |
|-----|---------|----------|------|
| **训练速度** | ~1.2s/iter | ~2.5s/iter | -2.1倍 |
| **显存占用** | ~15GB | ~28GB | +1.9倍 |
| **mAP（预期）** | 基准 | +2-3% | 分辨率优势 |
| **NDS（预期）** | 基准 | +1-2% | 检测质量提升 |

---

## 🔄 两种配置的完整参数对照表

| 参数名称 | 800×320 | 1600×640 | 变化倍率 |
|---------|---------|----------|---------|
| `voxel_size` | 0.1 | 0.075 | 0.75× |
| `img_backbone` | ResNet50 | VoVNet-99 | - |
| `final_dim` | (320, 800) | (640, 1600) | 2× |
| `pers_feature_shape` | (6, 20, 50) | (6, 40, 100) | 2× |
| `bev_feature_shape` | (128, 128) | (180, 180) | 1.41× |
| `camera_window` | 8 | 15 | 1.875× |
| `lidar_window` | 5 | 5 | 1× |
| `gaussian_sigma` | 1.0 | 2.0 | 2× |
| `window_size` | 8 | 15 | 1.875× |
| `lr` | 0.00014 | 0.0002 | 1.43× |
| `samples_per_gpu` | 1 | 1 | 1× |

---

## ⚠️ 注意事项

### **1. 特征图尺寸一致性检查**

确保以下配置中的特征图尺寸**完全一致**：
```python
# 这些地方必须保持相同的值
aqr_config.bev_feature_shape = renderer_config.bev_feature_shape = attention_bias_config.bev_feature_shape
aqr_config.pers_feature_shape = renderer_config.pers_feature_shape = attention_bias_config.pers_feature_shape
```

### **2. 窗口大小合理性**

```python
# window_size 应该小于特征图尺寸
window_size < min(pers_h, pers_w)  # 例如：15 < min(40, 100) ✅

# 但也不宜过大，影响局部性
window_size < pers_h / 2  # 推荐：15 < 40/2 = 20 ✅
```

### **3. 骨干网络匹配**

```python
# 800×320 → ResNet50（较轻量）
img_backbone=dict(type='ResNet', frozen_stages=4, ...)

# 1600×640 → VoVNet-99（更强大）
img_backbone=dict(type='VoVNet', spec_name='V-99-eSE', frozen_stages=4, ...)
```

---

## 🚀 快速切换指南

### **从800×320切换到1600×640**：

```bash
# 1. 切换配置文件
OLD: cmt_aqr_voxel0100_r50_800x320_cbgs.py
NEW: cmt_aqr_voxel0075_vov_1600x640_cbgs.py

# 2. 检查内存是否足够
nvidia-smi  # 需要至少28GB显存

# 3. 启动训练
python tools/train.py configs/fusion/cmt_aqr_voxel0075_vov_1600x640_cbgs.py
```

### **从1600×640降级到800×320**：

```bash
# 内存不足时可以降级
python tools/train.py configs/fusion/cmt_aqr_voxel0100_r50_800x320_cbgs.py
```

---

## 📝 配置文件清单

```
CMT-master/projects/configs/fusion/
├── cmt_aqr_voxel0100_r50_800x320_cbgs.py       # 🔥 800×320 AQR配置
├── cmt_aqr_voxel0075_vov_1600x640_cbgs.py      # 🔥 1600×640 AQR配置
├── cmt_voxel0100_r50_800x320_cbgs.py           # 800×320 Baseline
└── cmt_voxel0075_vov_1600x640_cbgs.py          # 1600×640 Baseline
```

---

**主人，现在您有了两套完整的AQR配置，可以根据GPU内存和精度需求灵活选择！** 🎉

