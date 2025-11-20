# 1600x640 分辨率 AQR 配置参数检查清单 ✅

## 📊 关键参数对照表

### 1. 特征图尺寸
| 参数 | 800x320 | 1600x640 | 配置位置 | 状态 |
|------|---------|----------|---------|------|
| **BEV特征图** | (128, 128) | **(180, 180)** | `bev_feature_shape` | ✅ |
| **透视特征图** | (6, 20, 50) | **(6, 40, 100)** | `pers_feature_shape` | ✅ |

计算依据：
- BEV: voxel_size=0.075, grid=1440, 1440/8=180
- Perspective: 1600/16=100(W), 640/16=40(H)

---

### 2. 窗口配置
| 参数 | 800x320 | 1600x640 | 说明 |
|------|---------|----------|------|
| **Camera窗口** | 8 | **15** | `window_sizes[0]` or `window_size` |
| **LiDAR窗口** | 5 | **5** | `window_sizes[1]` (保持不变) |
| **高斯Sigma** | 1.0-2.0 | **2.5** | `gaussian_sigma` |

窗口大小选择原则：
- Camera窗口约为特征图宽度的15-20% (100*0.15≈15)
- LiDAR保持5（BEV特征图更密集）

---

### 3. Attention Bias配置
| 参数 | 配置值 | 说明 | 状态 |
|------|--------|------|------|
| `window_size` | **15** | 局部窗口大小 | ✅ |
| `bias_scale` | **2.5** | 初始缩放因子 | ✅ |
| `learnable_scale` | **True** | 可学习scale | ✅ |
| `min_scale` | **0.5** | 最小scale | ✅ |
| `max_scale` | **5.0** | 最大scale | ✅ |
| `use_local_bias` | **True** | 使用局部bias | ✅ |
| `use_gaussian_window` | **False** | 高斯窗口（首次实验关闭） | ✅ |
| `gaussian_sigma` | **2.5** | 高斯标准差 | ✅ |
| `debug_print` | **True** | 调试打印 | ✅ |
| `print_interval` | **1000** | 打印间隔 | ✅ |
| `fp16` | **True** | FP16精度 | ✅ |

---

### 4. AQR权重生成器配置
| 参数 | 配置值 | 状态 |
|------|--------|------|
| `embed_dims` | **256** | ✅ |
| `window_sizes` | **[15, 5]** | ✅ [camera, lidar] |
| `use_type_embed` | **True** | ✅ |
| `num_layers` | **1** | ✅ |
| `num_heads` | **4** | ✅ |

---

### 5. 训练配置
| 参数 | 800x320 | 1600x640 | 状态 |
|------|---------|----------|------|
| **学习率** | 0.0001 | **0.0002** | ✅ |
| **可视化间隔** | 100 | **1000** | ✅ |
| **Backbone** | ResNet50 | **VoVNet-99** | ✅ |
| **frozen_stages** | 4 | **4** | ✅ |
| **norm_eval** | True | **True** | ✅ |

---

## 🔥 代码默认值检查

### attention_bias_generator.py
```python
✅ bev_feature_shape=(180, 180)       # 1600x640
✅ pers_feature_shape=(6, 40, 100)    # 1600x640
✅ window_size=15                     # 1600x640
✅ gaussian_sigma=2.5                 # 1600x640
✅ learnable_scale=True               # 推荐
✅ debug_print=True                   # 启用
✅ print_interval=1000                # 降低频率
✅ fp16=True                          # 节省内存
```

### 配置文件
```python
✅ cmt_aqr_voxel0075_vov_1600x640_cbgs.py
   - bev_feature_shape=(180, 180)
   - pers_feature_shape=(6, 40, 100)
   - window_size=15
   - window_sizes=[15, 5]
   - gaussian_sigma=2.5
   - _delete_=True (确保配置覆盖)
```

---

## 🎯 与800x320的关键差异

### 硬件差异
```
800x320:
- Backbone: ResNet50
- 特征图: BEV (128,128), Pers (6,20,50)
- 窗口: Camera=8, LiDAR=5
- 学习率: 0.0001

1600x640:
- Backbone: VoVNet-99 (更强)
- 特征图: BEV (180,180), Pers (6,40,100)
- 窗口: Camera=15, LiDAR=5
- 学习率: 0.0002 (更高)
```

### 计算量对比
```
800x320:
- BEV features: 128×128 = 16,384
- Perspective features: 6×20×50 = 6,000
- Total: 22,384 features
- Window coverage: 64 positions per query

1600x640:
- BEV features: 180×180 = 32,400 (+98%)
- Perspective features: 6×40×100 = 24,000 (+300%)
- Total: 56,400 features (+152%)
- Window coverage: 225 positions per query (+252%)
```

---

## ⚠️ 潜在问题检查清单

### 问题1: 配置传递失败
**症状**: 初始化信息显示错误的尺寸
**检查方法**:
```bash
# 查看训练开始的初始化信息
✅ AttentionBiasGenerator initialized:
   BEV shape: (180, 180)     ← 应该是这个
   Pers shape: (6, 40, 100)  ← 应该是这个
   Window size: 15            ← 应该是这个
```

**如果显示(128, 128)或window_size=8**:
- 配置文件传递失败
- 需要检查_delete_=True是否生效

### 问题2: 内存不足
**症状**: OOM (Out of Memory)
**解决方案**:
```python
# 减少batch_size
data = dict(
    samples_per_gpu=1  # 从2降到1
)

# 或启用梯度检查点
transformerlayers=dict(
    with_cp=True  # 启用checkpoint
)
```

### 问题3: 窗口大小过大
**症状**: 训练速度显著变慢
**解决方案**:
```python
# 尝试减小窗口
window_size=12  # 从15降到12
# 或
window_size=10  # 从15降到10
```

### 问题4: bias_scale不变
**症状**: 训练数千个iteration后scale仍是2.5000
**解决方案**:
```python
# 检查是否启用可学习
learnable_scale=True  # 确保为True

# 或增大学习率
'attention_bias_generator.bias_scale': dict(lr_mult=1.0)  # 从0.5增到1.0
```

---

## 📋 训练前最后检查

### Step 1: 清理缓存
```bash
rm -rf work_dirs/cmt_aqr_1600x640/.mim*
rm -rf work_dirs/cmt_aqr_1600x640/config.py
```

### Step 2: 验证配置
```bash
python tools/test_config_loading.py
```

预期输出:
```
✅ enable_aqr = True
✅ debug_mode = True
✅ bev_feature_shape = (180, 180)
✅ pers_feature_shape = (6, 40, 100)
✅ window_size = 15
```

### Step 3: 启动训练
```bash
# 8卡训练
python tools/train.py \
    projects/configs/fusion/cmt_aqr_voxel0075_vov_1600x640_cbgs.py \
    --work-dir work_dirs/cmt_aqr_1600x640 \
    --launcher pytorch \
    --deterministic \
    --seed 0
```

### Step 4: 观察初始化输出
```
预期看到:
✅ AttentionBiasGenerator initialized:
   BEV shape: (180, 180)      ← 正确！
   Pers shape: (6, 40, 100)   ← 正确！
   Window size: 15 (local)    ← 正确！
   Bias scale: 2.5 (learnable) ← 正确！
   Scale range: [0.5, 5.0]
   FP16: True
```

---

## 🎉 准备就绪！

所有参数已检查完毕，可以开始训练！

**建议的训练流程**:
1. 先训练1个epoch，验证配置正确
2. 检查Iteration 1000/3000的详细报告
3. 观察bias_scale是否开始变化
4. 对比800x320的性能差异

祝训练顺利！🐾✨

