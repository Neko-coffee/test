# Attention Bias 调试功能快速总结 ⚡

## ✅ 已完成的功能

### 1. **实时调试打印** 📊

在 `AttentionBiasGenerator` 中添加了完整的实时监控功能，每隔指定iteration自动打印：

- **Bias Scale**：当前缩放因子（learnable时显示变化）
- **Bias Statistics**：整体、BEV、Camera的统计信息（Mean、Std、Range）
- **Distribution**：正负比例、强bias比例、near-zero比例
- **Input Weights**：AQR生成的原始权重统计
- **Modality Preference**：模态偏好分析

---

## 🔧 如何使用

### **配置文件设置**

在 `cmt_aqr_voxel0100_r50_800x320_cbgs.py` 中：

```python
attention_bias_config=dict(
    type='AttentionBiasGenerator',
    # ... 其他配置 ...
    
    # 🔥 调试配置（新增）
    debug_print=True,        # 启用调试打印
    print_interval=100,      # 每100个iteration打印一次
)
```

---

## 📊 输出示例

训练时每100个iteration会看到：

```
======================================================================
📊 [AttentionBias] Monitoring Report (Iter 100):
   🔧 Learnable Bias Scale: 2.4532 (range: [0.5, 5.0])
   📈 Bias Statistics:
      Overall  - Mean: +0.0234, Std: 1.2456
                 Range: [-4.8234, +4.7891]
      BEV      - Mean: +0.0456, Std: 1.3021
      Camera   - Mean: +0.0012, Std: 1.1891
      Distribution:
         Positive: 52.3% | Negative: 47.7%
         Strong+ (>+2): 15.23% | Strong- (<-2): 13.45%
         Near-zero (|bias|<0.1): 8.9%
   📊 Input Weights (from AQR):
      LiDAR    - Mean: +0.1234, Std: 0.4521
                 Range: [-0.9234, +0.8912]
      Camera   - Mean: -0.0456, Std: 0.3891
                 Range: [-0.8734, +0.9123]
      Modality Preference:
         LiDAR-preferred: 58.3% | Camera-preferred: 41.7%
         Balanced (diff<0.2): 23.4%
======================================================================
```

---

## ⚠️ 需要关注的异常信号

| 异常信号 | 正常范围 | 说明 |
|---------|---------|------|
| Scale接近极限 | `[1.0, 4.0]` | >4.5或<0.55时需要调整 |
| Positive比例失衡 | `[40%, 60%]` | >80%或<20%表示严重偏向 |
| Strong bias过多 | `[10%, 30%]` | >50%可能影响稳定性 |
| Near-zero过多 | `[5%, 15%]` | >50%说明AQR权重生成失效 |
| LiDAR偏好极端 | `[30%, 70%]` | >90%或<10%可能退化为单模态 |

---

## 🎯 推荐使用策略

### **训练初期（前500个iter）**
```python
debug_print=True,
print_interval=50,   # 更频繁监控
```

### **训练中期（稳定后）**
```python
debug_print=True,
print_interval=200,  # 降低打印频率
```

### **生产环境**
```python
debug_print=False,   # 关闭调试打印
```

---

## 📝 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `attention_bias_generator.py` | 添加 `debug_print`、`print_interval` 参数和完整的监控逻辑 |
| `cmt_aqr_voxel0100_r50_800x320_cbgs.py` | 配置中启用 `debug_print=True, print_interval=100` |

---

## 🚀 立即开始

修改后的文件已经配置好 `debug_print=True`，**直接运行训练即可看到实时监控输出**！

---

**主人，现在您可以实时监控Attention Bias的运行状态了！** 🎉

