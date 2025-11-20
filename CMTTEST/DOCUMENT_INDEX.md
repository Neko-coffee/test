# CMT-AQR 项目文档索引 📚

> 📅 **最后更新时间：** 2024-10-16  
> 📊 **文档总数：** 30个  
> 🎯 **项目状态：** 🎉 Attention Bias方案实现完成 + 调试功能集成 + 双分辨率配置！

---

## 📑 目录

- [🚀 快速导航](#快速导航)
- [📊 实验分析文档](#实验分析文档)
- [🔧 技术方案文档](#技术方案文档)
- [🐛 问题诊断文档](#问题诊断文档)
- [📖 使用指南文档](#使用指南文档)
- [📝 项目管理文档](#项目管理文档)

---

## 🚀 快速导航

### 最新重要文档（优先查看）⭐

| 文档 | 描述 | 重要程度 | 创建时间 |
|-----|------|---------|---------|
| [BACKBONE_FREEZE_STRATEGY.md](#骨干网络冻结策略) | ❄️ 骨干网络冻结策略分析和实施指南 | ⭐⭐⭐⭐⭐ | 2024-10-13 |
| [SCALE_CONSTRAINT_IMPLEMENTATION_SUMMARY.md](#scale约束实现总结) | ✅ Scale约束完整实现总结 | ⭐⭐⭐⭐⭐ | 2024-10-13 |
| [SOFTMAX_SENSITIVITY_AND_SCALE_CONSTRAINT.md](#softmax敏感区间与scale约束) | 🔥 Softmax敏感区间分析和Scale约束策略 | ⭐⭐⭐⭐⭐ | 2024-10-13 |
| [LEARNABLE_BIAS_SCALE.md](#可学习的bias-scale) | 🎓 可学习的bias_scale机制 | ⭐⭐⭐⭐⭐ | 2024-10-13 |
| [LOCAL_ATTENTION_BIAS_IMPLEMENTATION_DESIGN.md](#局部attention-bias实现设计) | 🚀 局部Attention Bias完整实现方案 | ⭐⭐⭐⭐⭐ | 2024-10-13 |
| [ATTENTION_BIAS_THREE_QUESTIONS_EXPLAINED.md](#attention-bias核心问题解答) | 💡 Attention Bias三大问题详解 | ⭐⭐⭐⭐⭐ | 2024-10-13 |
| [QUERY_LEVEL_VS_LOCAL_WINDOW_BIAS_EXPLAINED.md](#query-level与local-window-bias对比) | 🔍 Query-level vs Local Window bias区别 | ⭐⭐⭐⭐⭐ | 2024-10-13 |
| [AQR_ATTENTION_BIAS_MODULATION_PROPOSAL.md](#1-aqr_attention_bias_modulation_proposalmd) | 🔥 Attention Bias调制方案（推荐方案） | ⭐⭐⭐⭐⭐ | 2024-10-13 |
| [AQR_SECONDARY_NETWORK_MODULATION_PROPOSAL.md](#2-aqr_secondary_network_modulation_proposalmd) | 🧠 二次网络调制方案（备选方案） | ⭐⭐⭐⭐ | 2024-10-13 |
| [AQR_FINAL_CONCLUSIONS.md](#3-aqr_final_conclusionsmd) | 📊 三次实验完整总结和最终结论 | ⭐⭐⭐⭐⭐ | 2024-10-13 |

### 核心实验报告

| 文档 | 描述 | 状态 |
|-----|------|-----|
| [AQR_THREE_EXPERIMENTS_COMPLETE_COMPARISON.md](#6-aqr_three_experiments_complete_comparisonmd) | 三次实验完整对比（1.5+0.7, 2.5+0.6, Baseline） | ✅ 完成 |
| [AQR_800x320_TRAINING_ANALYSIS.md](#7-aqr_800x320_training_analysismd) | 第一次AQR训练分析（1.5+0.7） | ✅ 完成 |
| [AQR_2.5_0.6_TRAINING_ANALYSIS.md](#8-aqr_25_06_training_analysismd) | 第二次AQR训练分析（2.5+0.6） | ✅ 完成 |

---

## 📊 实验分析文档

### 1. AQR_ATTENTION_BIAS_MODULATION_PROPOSAL.md
**📄 类型：** 技术方案设计  
**🎯 目的：** 提出通过Attention Bias调制来替代直接特征调制的新方案  
**📝 内容：**
- 理论基础：不改变特征值，只调制attention权重
- 详细设计：AttentionBiasGenerator、集成方案
- 优势分析：完全兼容Flash Attention，不破坏特征分布
- 实现计划：分4个阶段，预计总工时6-8小时

**🔑 关键亮点：**
- ✅ 理论最稳健（保护预训练分布）
- ✅ 实现最简单（约150行代码）
- ✅ 0额外参数
- ✅ 完全兼容现有架构

**📍 适用场景：** 优先尝试，理论风险最低

---

### 2. AQR_SECONDARY_NETWORK_MODULATION_PROPOSAL.md
**📄 类型：** 技术方案设计  
**🎯 目的：** 提出用小型网络重对齐调制后特征的方案  
**📝 内容：**
- 核心思想：调制 → 网络修正 → 对齐特征
- 三种实现方案：通道注意力、深度可分离卷积、AdaIN
- 参数量分析：约73K参数（<0.2%总参数）
- 理论分析：域适应、分布对齐、残差学习

**🔑 关键亮点：**
- ✅ 保留AQR自适应能力
- ✅ 尝试修复分布破坏
- ✅ 轻量级设计
- ✅ 可插拔模块

**📍 适用场景：** 如果Attention Bias效果不够好时的备选方案

---

### 3. AQR_FINAL_CONCLUSIONS.md
**📄 类型：** 实验总结  
**🎯 目的：** 三次实验的完整总结和最终结论  
**📝 内容：**
- 三次实验对比表格（性能、参数、现象）
- 5个核心问题的回答
- AQR失败的根本原因分析
- 未来方向建议

**🔑 关键发现：**
- ❌ 特征调制破坏预训练分布（核心问题）
- ❌ 小目标性能下降严重
- ✅ 大目标相对稳定
- 💡 应该改为注意力调制

**📍 适用场景：** 了解项目当前状态和结论

---

### 4. MOME_DN_CORRECTION.md
**📄 类型：** 重要更正  
**🎯 目的：** 更正之前关于MoME不使用DN的错误判断  
**📝 内容：**
- 证据：MoME确实使用DN，配置完全相同
- CMT vs MoME DN参数对比
- 真正的差异分析：多专家机制 vs 权重图渲染
- 对AQR失败原因的重新理解

**🔑 关键洞察：**
- ✅ DN不是问题（MoME也用）
- ❌ 问题在于特征调制方式
- 💡 MoME在Transformer内部做路由（优雅）
- 💡 我们在Transformer外部做调制（粗暴）

**📍 适用场景：** 理解为什么DN不是问题所在

---

### 5. AQR_5_CRITICAL_QUESTIONS_ANALYSIS.md
**📄 类型：** 深度分析  
**🎯 目的：** 回答5个核心问题，深入理解AQR失败原因  
**📝 内容：**
- Q1: DN是否影响AQR性能
- Q2: 直接乘特征是否有问题
- Q3: 为什么小目标下降严重
- Q4: 为什么大目标相对稳定
- Q5: 未来如何改进

**🔑 关键结论：**
- DN不是问题（MoME验证）
- 特征调制破坏分布是核心问题
- 小目标对特征分布更敏感
- 应该改为注意力调制

**📍 适用场景：** 深入理解技术细节和失败原因

---

### 6. AQR_THREE_EXPERIMENTS_COMPLETE_COMPARISON.md
**📄 类型：** 实验对比  
**🎯 目的：** 详细对比三次实验的结果  
**📝 内容：**
- 实验1：1.5+0.7（第一次AQR）
- 实验2：2.5+0.6（第二次AQR）
- 实验3：Baseline（无AQR）
- 性能对比、参数对比、现象对比

**🔑 关键数据：**
```
Baseline: 67.9% mAP (预训练) → 67.4% (1 epoch)
AQR 1.5+0.7: 67.9% → 65.8% (↓1.6%)
AQR 2.5+0.6: 67.9% → 65.1% (↓2.3%)
```

**📍 适用场景：** 查看详细的实验数据对比

---

### 7. AQR_800x320_TRAINING_ANALYSIS.md
**📄 类型：** 训练分析  
**🎯 目的：** 第一次AQR训练的详细分析  
**📝 内容：**
- 性能指标分析（mAP: 65.8%, NDS: 69.2%）
- 训练稳定性分析（loss、梯度）
- AQR权重分布分析
- 特征调制效果分析
- 每个类别性能分析

**🔑 重要说明：**
- ⚠️ 这是微调训练（不是从零开始）
- ⚠️ 包含DN训练的1730个query
- ⚠️ Camera权重裁剪bug已修复

**📍 适用场景：** 了解第一次AQR实验的详细情况

---

### 8. AQR_2.5_0.6_TRAINING_ANALYSIS.md
**📄 类型：** 训练分析  
**🎯 目的：** 第二次AQR训练（更宽松参数）的分析  
**📝 内容：**
- 配置：max_weight_clamp=2.5, residual_weight=0.6
- 性能对比：65.1% mAP（比1.5+0.7更差）
- 权重分布分析
- 特征调制分析

**🔑 关键发现：**
- 放宽限制反而更差
- 证明了过度调制的危害
- 需要从根本上改变方案

**📍 适用场景：** 了解参数调优的尝试结果

---

### 9. AQR_THREE_EXPERIMENTS_COMPARISON.md
**📄 类型：** 实验对比（已废弃）  
**🎯 目的：** 初步对比三次实验  
**⚠️ 状态：** 已被`AQR_THREE_EXPERIMENTS_COMPLETE_COMPARISON.md`替代  
**📍 适用场景：** 建议查看新版本

---

### 10. AQR_VS_BASELINE_COMPARISON.md
**📄 类型：** 对比分析（已废弃）  
**🎯 目的：** AQR与Baseline的对比  
**⚠️ 状态：** 信息已整合到最终结论文档  
**📍 适用场景：** 建议查看最终结论

---

## 🔧 技术方案文档

### 局部Attention Bias实现设计
**📄 文件：** LOCAL_ATTENTION_BIAS_IMPLEMENTATION_DESIGN.md  
**📄 类型：** 实现设计方案  
**🎯 目的：** 局部Attention Bias的完整实现设计和架构说明  
**📝 内容：**
- 窗口大小设计（物理空间分析、策略对比）
- 架构设计（流程图、核心模块）
- AttentionBiasGenerator模块详细设计
- 向量化实现算法
- Transformer集成点设计
- 性能预估和优化建议
- 配置接口设计
- 实现检查清单

**🔑 关键设计决策：**
```python
# 推荐配置
window_size = 15              # 与LAM的camera window一致
bias_scale = 1.0              # 初始无缩放
use_local_bias = True         # 启用局部bias
fp16 = True                   # 节省内存
```

**🏗️ 核心架构：**
```
AQRWeightGenerator → AttentionBiasGenerator → Transformer（添加bias）
   ↓                      ↓                         ↓
[bs,900] 权重      [bs,900,56400] bias      attention scores + bias
```

**📍 适用场景：** 实现局部Attention Bias功能时的完整参考

---

### Attention Bias核心问题解答
**📄 文件：** ATTENTION_BIAS_THREE_QUESTIONS_EXPLAINED.md  
**📄 类型：** 技术解释  
**🎯 目的：** 回答Attention Bias的三个核心问题  
**📝 内容：**
- Q1: Bias一般做法和影响
- Q2: Bias是否细粒度（per-query）
- Q3: 目标检测中的类似方法

**🔑 关键结论：**
- Linear bias ≠ Attention bias（完全不同）
- Attention bias是fine-grained的（每个query不同）
- DN-DETR、Relative Position Bias等成功案例

**📍 适用场景：** 理解Attention Bias的技术细节

---

### Query-level与Local Window Bias对比
**📄 文件：** QUERY_LEVEL_VS_LOCAL_WINDOW_BIAS_EXPLAINED.md  
**📄 类型：** 概念对比  
**🎯 目的：** 区分Query-level和Local Window两种bias方式  
**📝 内容：**
- Query-level bias：每个query有不同的bias值
- Local Window bias：bias在空间上局部化
- 二者区别和实现复杂度对比

**🔑 核心区别：**
```
Query-level：每个Q的bias都不同（全局施加）
Local Window：只在Q投影的局部窗口施加bias（空间局部化）

最终选择：Local Window bias（更符合空间先验）
```

**📍 适用场景：** 理解局部bias的设计理念

---

### 11. AQR_WEIGHT_RENDERING_INTEGRATION.md
**📄 类型：** 集成方案  
**🎯 目的：** AQR权重图渲染机制的集成文档  
**📝 内容：**
- AQR权重生成器设计
- 权重图渲染器设计
- 特征调制器设计
- 集成到CMT的完整流程

**🔑 关键组件：**
- AQRWeightGenerator
- WeightRenderer（高斯渲染、双线性插值）
- FeatureModulator（简单/完整模式）

**📍 适用场景：** 了解当前AQR实现的技术细节

---

### 12. AQR_PARAMETER_TUNING_GUIDE.md
**📄 类型：** 调优指南  
**🎯 目的：** AQR参数调优建议  
**📝 内容：**
- 关键参数说明（max_weight_clamp、residual_weight等）
- 调优策略建议
- 预期效果分析

**🔑 关键参数：**
- max_weight_clamp: 权重图裁剪上限
- residual_weight: 残差连接权重
- gaussian_sigma: 高斯核标准差

**📍 适用场景：** 进行参数调优时参考（但现在方案可能要改）

---

### 13. CMT_DN_PARAMETERS_LOCATION.md
**📄 类型：** 代码定位  
**🎯 目的：** 记录CMT DN相关参数的具体位置  
**📝 内容：**
- DN参数位置：cmt_head.py 216-220行
- DN实现逻辑：prepare_for_dn函数
- DN损失计算：_dn_loss_single_task函数

**🔑 DN参数：**
- scalar=10
- noise_scale=1.0
- noise_trans=0.0
- dn_weight=1.0
- split=0.75

**📍 适用场景：** 需要修改DN相关代码时查找位置

---

### 14. DEPLOYMENT_GUIDE.md
**📄 类型：** 部署指南  
**🎯 目的：** AQR-CMT的部署说明  
**📝 内容：**
- 环境配置
- 数据准备
- 训练命令
- 测试命令

**📍 适用场景：** 新环境部署或新成员入门

---

## 🐛 问题诊断文档

### 15. AQR_QUERY_AND_CAMERA_BUG_ANALYSIS.md
**📄 类型：** Bug分析  
**🎯 目的：** 分析Query数量和Camera权重裁剪bug  
**📝 内容：**
- Query数量1730的原因（DN训练）
- Camera max weight 70.83的bug
- Bug修复方案

**🔑 Bug修复：**
```python
# 在weight_renderer.py添加：
for view_idx in range(num_views):
    weight_map[:, view_idx] = self._postprocess_weight_map(weight_map[:, view_idx])
```

**📍 适用场景：** 了解已修复的bug

---

### 16. DDP_CHECKPOINT_ERROR_FIX.md
**📄 类型：** Bug修复  
**🎯 目的：** 修复分布式训练时的checkpoint加载错误  
**📝 内容：**
- 错误现象：RuntimeError: Error(s) in loading state_dict
- 原因分析：DDP包装导致的key不匹配
- 修复方案：strip_prefix_if_present

**📍 适用场景：** 遇到类似DDP错误时参考

---

### 17. AQR_TRAINING_ISSUE_ANALYSIS.md
**📄 类型：** 问题分析  
**🎯 目的：** 早期AQR训练问题的分析  
**📝 内容：**
- 训练不稳定现象
- 可能原因分析
- 初步解决建议

**⚠️ 状态：** 早期文档，部分信息已过时  
**📍 适用场景：** 历史参考

---

## 📖 使用指南文档

### 18. AQR_METRICS_EXPLANATION.md
**📄 类型：** 指标说明  
**🎯 目的：** 解释训练日志中的各项指标  
**📝 内容：**
- cameralidar_weights_stats（mean、std、min、max）
- modulation_effect（mean_change、std_change、relative_change）
- query数量1730的解释

**🔑 关键指标：**
```
mean: 所有query权重的平均值
std: 权重的标准差（分散程度）
relative_change: 相对变化率 = |调制后-调制前| / 调制前
```

**📍 适用场景：** 理解训练日志输出

---

### 19. AQR_FEATURE_CALCULATION_EXAMPLE.md
**📄 类型：** 计算示例  
**🎯 目的：** 用具体数字展示特征调制的计算过程  
**📝 内容：**
- max_weight_clamp=2.0, residual_weight=0.6时的计算
- 特征图最大值的计算示例
- 可视化说明

**📍 适用场景：** 理解特征调制的数值变化

---

### 20. AQR_BASELINE_COMPARISON_GUIDE.md
**📄 类型：** 实验指南  
**🎯 目的：** 指导如何进行Baseline对比实验  
**📝 内容：**
- 创建Baseline配置文件
- 运行Baseline训练
- 对比分析方法

**📍 适用场景：** 进行对比实验时参考

---

## 📝 项目管理文档

### 21. AQR_CMT_CHANGELOG.md
**📄 类型：** 变更日志  
**🎯 目的：** 记录AQR-CMT项目的所有变更  
**📝 内容：**
- 版本历史
- 功能更新
- Bug修复记录

**📍 适用场景：** 查看项目历史变更

---

### 22. AQR_THREE_QUESTIONS_SUMMARY.md
**📄 类型：** 问题总结  
**🎯 目的：** 总结项目中的三个关键问题  
**📝 内容：**
- 问题1：AQR性能下降
- 问题2：参数调优方向
- 问题3：下一步方案

**⚠️ 状态：** 已被更全面的5问题分析替代  
**📍 适用场景：** 历史参考

---

### 23. README.md
**📄 类型：** 项目说明  
**🎯 目的：** 项目整体介绍  
**📝 内容：**
- 项目概述
- 快速开始
- 目录结构
- 使用说明

**📍 适用场景：** 项目入门第一站

---

## 🗂️ 文档分类索引

### 按重要程度

#### ⭐⭐⭐⭐⭐ 必读文档
1. AQR_ATTENTION_BIAS_MODULATION_PROPOSAL.md
2. AQR_SECONDARY_NETWORK_MODULATION_PROPOSAL.md
3. AQR_FINAL_CONCLUSIONS.md
4. MOME_DN_CORRECTION.md

#### ⭐⭐⭐⭐ 重要参考
5. AQR_5_CRITICAL_QUESTIONS_ANALYSIS.md
6. AQR_THREE_EXPERIMENTS_COMPLETE_COMPARISON.md
7. AQR_800x320_TRAINING_ANALYSIS.md
8. AQR_WEIGHT_RENDERING_INTEGRATION.md

#### ⭐⭐⭐ 一般参考
9. AQR_2.5_0.6_TRAINING_ANALYSIS.md
10. AQR_QUERY_AND_CAMERA_BUG_ANALYSIS.md
11. CMT_DN_PARAMETERS_LOCATION.md
12. AQR_METRICS_EXPLANATION.md

#### ⭐⭐ 可选阅读
13. AQR_PARAMETER_TUNING_GUIDE.md
14. AQR_FEATURE_CALCULATION_EXAMPLE.md
15. DEPLOYMENT_GUIDE.md

#### ⭐ 历史存档
16. AQR_THREE_EXPERIMENTS_COMPARISON.md
17. AQR_VS_BASELINE_COMPARISON.md
18. AQR_TRAINING_ISSUE_ANALYSIS.md
19. AQR_THREE_QUESTIONS_SUMMARY.md

---

### 按主题分类

#### 🎯 实验结果
- AQR_800x320_TRAINING_ANALYSIS.md
- AQR_2.5_0.6_TRAINING_ANALYSIS.md
- AQR_THREE_EXPERIMENTS_COMPLETE_COMPARISON.md
- AQR_FINAL_CONCLUSIONS.md

#### 🔬 问题分析
- AQR_5_CRITICAL_QUESTIONS_ANALYSIS.md
- MOME_DN_CORRECTION.md
- AQR_QUERY_AND_CAMERA_BUG_ANALYSIS.md
- AQR_TRAINING_ISSUE_ANALYSIS.md

#### 💡 技术方案
- AQR_ATTENTION_BIAS_MODULATION_PROPOSAL.md
- AQR_SECONDARY_NETWORK_MODULATION_PROPOSAL.md
- AQR_WEIGHT_RENDERING_INTEGRATION.md
- AQR_PARAMETER_TUNING_GUIDE.md

#### 📚 使用指南
- AQR_METRICS_EXPLANATION.md
- AQR_FEATURE_CALCULATION_EXAMPLE.md
- AQR_BASELINE_COMPARISON_GUIDE.md
- DEPLOYMENT_GUIDE.md

#### 🐛 Bug修复
- AQR_QUERY_AND_CAMERA_BUG_ANALYSIS.md
- DDP_CHECKPOINT_ERROR_FIX.md

#### 📋 项目管理
- README.md
- AQR_CMT_CHANGELOG.md
- CMT_DN_PARAMETERS_LOCATION.md

---

## 🔄 文档更新记录

### 2024-10-13
- ✅ 新增：AQR_ATTENTION_BIAS_MODULATION_PROPOSAL.md
- ✅ 新增：AQR_SECONDARY_NETWORK_MODULATION_PROPOSAL.md
- ✅ 新增：MOME_DN_CORRECTION.md
- ✅ 新增：AQR_5_CRITICAL_QUESTIONS_ANALYSIS.md
- ✅ 更新：AQR_FINAL_CONCLUSIONS.md（完整版）
- ✅ 新增：本索引文档 DOCUMENT_INDEX.md

### 2024-10-12
- ✅ 新增：AQR_THREE_EXPERIMENTS_COMPLETE_COMPARISON.md
- ✅ 新增：CMT_DN_PARAMETERS_LOCATION.md
- ✅ 修复：AQR_800x320_TRAINING_ANALYSIS.md（微调说明）

### 更早记录
- 见各文档内部的修改历史

---

## 📊 文档统计

```
总文档数：23个
├─ 实验分析：8个
├─ 技术方案：4个
├─ 问题诊断：4个
├─ 使用指南：4个
└─ 项目管理：3个

代码量（MD文档）：
├─ 方案设计：~1700行（Attention Bias + 二次网络）
├─ 实验分析：~2000行（各次实验报告）
├─ 问题诊断：~1500行（5问题分析 + DN更正）
└─ 其他文档：~1500行

总计：约6700行详细文档
```

---

## 🎯 推荐阅读路径

### 路径1：快速了解项目现状（30分钟）
```
1. README.md（5分钟）
   ↓
2. AQR_FINAL_CONCLUSIONS.md（10分钟）
   ↓
3. AQR_ATTENTION_BIAS_MODULATION_PROPOSAL.md（15分钟）
```

### 路径2：深入理解技术细节（2小时）
```
1. AQR_WEIGHT_RENDERING_INTEGRATION.md（30分钟）
   ↓
2. AQR_800x320_TRAINING_ANALYSIS.md（30分钟）
   ↓
3. AQR_5_CRITICAL_QUESTIONS_ANALYSIS.md（30分钟）
   ↓
4. MOME_DN_CORRECTION.md（15分钟）
   ↓
5. AQR_ATTENTION_BIAS_MODULATION_PROPOSAL.md（15分钟）
```

### 路径3：准备实施新方案（1小时）
```
1. AQR_FINAL_CONCLUSIONS.md（15分钟）
   ↓
2. AQR_ATTENTION_BIAS_MODULATION_PROPOSAL.md（30分钟）
   ↓
3. AQR_SECONDARY_NETWORK_MODULATION_PROPOSAL.md（15分钟）
```

### 路径4：调试和问题排查（按需）
```
遇到问题时：
1. 先查看 AQR_METRICS_EXPLANATION.md（理解指标）
2. 查看对应的问题诊断文档
3. 参考 AQR_QUERY_AND_CAMERA_BUG_ANALYSIS.md（已知bug）
```

### 路径5：实施Attention Bias方案（推荐）
```
1. AQR_ATTENTION_BIAS_CORRECT_IMPLEMENTATION.md（30分钟）
   ↓
2. AQR_ATTENTION_BIAS_IMPLEMENTATION_COMPLETE.md（20分钟）
   ↓
3. 运行 tools/test_attention_bias_integration.py（5分钟）
   ↓
4. 开始训练实验（数小时）
```

---

## 📄 最新文档（2024-10-13）

### 26. AQR_ATTENTION_BIAS_CORRECT_IMPLEMENTATION.md
**📄 类型：** 技术实现  
**🎯 目的：** 基于同学提供的伪代码，给出正确的CMT架构适配方案  
**📝 内容：**
- 伪代码分析（正确和需修正的部分）
- CMT架构下的正确实现
- Key/Value融合处理
- Bias生成和应用
- 与伪代码的对比

**🔑 核心修正：**
- CMT中memory已融合，不需要concat
- 使用局部窗口bias，而非全局repeat
- 正确的维度转换和多头扩展

**📍 适用场景：** 理解AQR如何适配CMT架构

---

### 27. AQR_ATTENTION_BIAS_IMPLEMENTATION_COMPLETE.md
**📄 类型：** 实现总结  
**🎯 目的：** Attention Bias方案的完整实现总结和使用指南  
**📝 内容：**
- 实现清单（7个组件全部完成）
- 关键修改点详解
- 完整数据流图
- 新旧方案对比
- 使用方法和配置示例
- 性能预期和后续工作

**🔑 核心成就：**
```
✅ AttentionBiasGenerator - 局部窗口bias生成
✅ PETRMultiheadAttention - 支持float attn_mask
✅ CmtTransformer - 参数传递
✅ CmtHead - 完整集成
✅ 配置文件 - attention_bias_config
✅ 测试脚本 - 验证集成
```

**📍 适用场景：** 快速了解Attention Bias方案的完整实现

---

### 28. ATTN_MASK_FLOAT_MODE_VERIFICATION.md
**📄 类型：** 技术验证  
**🎯 目的：** 验证PyTorch MultiheadAttention支持float类型attn_mask  
**📝 内容：**
- PyTorch官方文档验证
- 网络资源验证
- 实践验证代码
- 实现简化分析

**🔑 核心发现：**
```python
# ✅ PyTorch原生支持！
attn_mask: FloatTensor  # 直接加到attention scores
# 无需修改MultiheadAttention内部
# 大大简化了实现
```

**📍 适用场景：** 理解float attn_mask的技术细节

---

## 💡 使用建议

### 📌 文档命名规范
```
AQR_<主题>_<类型>.md

示例：
- AQR_ATTENTION_BIAS_MODULATION_PROPOSAL.md
  ├─ AQR: 项目前缀
  ├─ ATTENTION_BIAS_MODULATION: 主题
  └─ PROPOSAL: 类型（方案设计）

常用类型：
- ANALYSIS: 分析报告
- PROPOSAL: 方案设计
- GUIDE: 使用指南
- COMPARISON: 对比分析
- FIX: Bug修复
```

### 📌 查找技巧
```
1. 按时间：看"最新重要文档"表格
2. 按主题：看"按主题分类"索引
3. 按重要性：看"按重要程度"索引
4. 模糊搜索：用文件名关键词搜索
```

### 📌 文档维护
```
每次创建新文档时：
1. ✅ 更新 DOCUMENT_INDEX.md
2. ✅ 添加到对应分类
3. ✅ 更新文档统计
4. ✅ 记录到更新日志
```

---

## 🔗 相关资源

### 代码仓库
- CMT主仓库：`CMT-master/`
- MoME参考仓库：`MoME-main/`

### 配置文件
- AQR配置：`projects/configs/fusion/cmt_aqr_voxel0100_r50_800x320_cbgs.py`
- Baseline配置：`projects/configs/fusion/cmt_baseline_voxel0100_r50_800x320_cbgs.py`

### 日志文件
- 第一次AQR：`AQRCMT800_320.log`
- 第二次AQR：`_data_coding_AQRCMT800_2.5_0.6.log`
- Baseline：`_data_coding_WOAQRCMT800.log`

---

## 📝 最新文档详细说明

### Softmax敏感区间与Scale约束

**文件**: `SOFTMAX_SENSITIVITY_AND_SCALE_CONSTRAINT.md`

**核心内容**:
- 📐 Softmax数学分析和敏感区间推导
- 🔬 不同bias值下的softmax响应曲线
- ⚙️ 三种约束策略：硬约束(Clamp)、软约束(L2正则)、参数化约束(Sigmoid)
- 📊 实验验证不同scale的饱和程度
- 📋 推荐配置：`min_scale=0.5, max_scale=5.0`

**关键发现**:
| Score差值 | Attention分布 | 状态 |
|----------|--------------|------|
| [-2, +2] | [0.12, 0.88] | ✅ 敏感区间 |
| [-5, +5] | [0.007, 0.993] | ⚠️ 接近饱和 |
| [-10, +10] | [0.00005, 0.99995] | ❌ 完全饱和 |

### 可学习的Bias Scale

**文件**: `LEARNABLE_BIAS_SCALE.md`

**核心内容**:
- 🎯 从固定scale到可学习scale的改进
- 🔧 使用`nn.Parameter`实现可学习参数
- 📈 预期的scale学习曲线
- 🎓 梯度传播和自适应机制分析
- 🔍 监控和调试方法

**实现方式**:
```python
if learnable_scale:
    self.bias_scale = nn.Parameter(torch.tensor(2.5))
else:
    self.register_buffer('bias_scale', torch.tensor(2.5))
```

**预期效果**:
- mAP提升：+0.2~0.5%
- 训练稳定性：⬆️
- 泛化能力：⬆️

---

**🐾 主人，这份索引会随着新文档的创建而持续更新！**

每次生成新MD文档时，我都会：
1. ✅ 更新这个索引
2. ✅ 添加新文档到对应分类
3. ✅ 更新统计信息
4. ✅ 记录更新日志

这样您随时都能快速找到需要的文档！📚✨

