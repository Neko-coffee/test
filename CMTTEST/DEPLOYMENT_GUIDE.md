# AQR-CMT 部署指南 🐾

本指南说明如何将AQR功能部署到原始CMT项目中。

## 📋 前置条件

- 原始CMT项目路径：`/path/to/original/CMT/`
- AQR-CMT项目路径：`/path/to/AQRCMT/CMT-master/`

## 🔧 部署步骤

### Step 1: 复制新增文件

```bash
# 设置路径变量
ORIGINAL_CMT="/path/to/original/CMT"
AQR_CMT="/path/to/AQRCMT/CMT-master"

# 1. 复制AQR核心模块（3个文件）
cp ${AQR_CMT}/projects/mmdet3d_plugin/models/utils/aqr_weight_generator.py \
   ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/utils/

cp ${AQR_CMT}/projects/mmdet3d_plugin/models/utils/weight_renderer.py \
   ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/utils/

cp ${AQR_CMT}/projects/mmdet3d_plugin/models/utils/feature_modulator.py \
   ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/utils/

# 2. 复制AQR配置文件
cp ${AQR_CMT}/projects/configs/fusion/cmt_aqr_voxel0075_vov_1600x640_cbgs.py \
   ${ORIGINAL_CMT}/projects/configs/fusion/

# 3. 复制文档（可选）
mkdir -p ${ORIGINAL_CMT}/.cursor/rules
cp ${AQR_CMT}/.cursor/rules/*.mdc ${ORIGINAL_CMT}/.cursor/rules/
```

### Step 2: 替换修改文件

```bash
# ⚠️ 建议先备份原文件
cp ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/dense_heads/cmt_head.py \
   ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/dense_heads/cmt_head_backup.py

cp ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/utils/__init__.py \
   ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/utils/__init_backup.py

# 替换修改后的文件
cp ${AQR_CMT}/projects/mmdet3d_plugin/models/dense_heads/cmt_head.py \
   ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/dense_heads/

cp ${AQR_CMT}/projects/mmdet3d_plugin/models/utils/__init__.py \
   ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/utils/
```

### Step 3: 验证部署

```bash
cd ${ORIGINAL_CMT}

# 检查文件是否存在
echo "🔍 检查AQR模块文件..."
ls -lh projects/mmdet3d_plugin/models/utils/aqr_weight_generator.py
ls -lh projects/mmdet3d_plugin/models/utils/weight_renderer.py
ls -lh projects/mmdet3d_plugin/models/utils/feature_modulator.py

echo "🔍 检查配置文件..."
ls -lh projects/configs/fusion/cmt_aqr_voxel0075_vov_1600x640_cbgs.py

echo "🔍 检查修改后的文件..."
grep -n "enable_aqr" projects/mmdet3d_plugin/models/dense_heads/cmt_head.py
grep -n "aqr_weight_generator" projects/mmdet3d_plugin/models/utils/__init__.py
```

### Step 4: 测试部署结果

```bash
# 4.1 测试标准CMT（不启用AQR）
python tools/train.py projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py

# 4.2 测试AQR-CMT（启用AQR）
python tools/train.py projects/configs/fusion/cmt_aqr_voxel0075_vov_1600x640_cbgs.py
```

## 📁 部署后的文件结构

```
original/CMT/
├── projects/
│   ├── mmdet3d_plugin/
│   │   ├── models/
│   │   │   ├── dense_heads/
│   │   │   │   ├── cmt_head.py              # ✏️ 已修改（集成AQR）
│   │   │   │   └── cmt_head_backup.py       # 📦 备份
│   │   │   └── utils/
│   │   │       ├── __init__.py              # ✏️ 已修改（导入AQR）
│   │   │       ├── __init_backup.py         # 📦 备份
│   │   │       ├── aqr_weight_generator.py  # 🆕 新增
│   │   │       ├── weight_renderer.py       # 🆕 新增
│   │   │       ├── feature_modulator.py     # 🆕 新增
│   │   │       ├── cmt_transformer.py       # 原有
│   │   │       └── petr_transformer.py      # 原有
│   └── configs/
│       └── fusion/
│           ├── cmt_voxel0075_vov_1600x640_cbgs.py     # 原有（标准CMT）
│           └── cmt_aqr_voxel0075_vov_1600x640_cbgs.py # 🆕 新增（AQR-CMT）
└── .cursor/
    └── rules/
        ├── aqr-configuration-guide.mdc              # 🆕 新增
        ├── aqr-debug-and-troubleshooting.mdc        # 🆕 新增
        ├── aqr-implementation-standards.mdc         # 🆕 新增
        ├── cmt-data-pipeline-guide.mdc              # 🆕 新增
        ├── cmt-feature-extraction-guide.mdc         # 🆕 新增
        ├── cmt-transformer-fusion-guide.mdc         # 🆕 新增
        ├── cmt-detection-head-guide.mdc             # 🆕 新增
        ├── cmt-training-inference-guide.mdc         # 🆕 新增
        ├── cmt-configuration-system-guide.mdc       # 🆕 新增
        └── cmt-project-overview.mdc                 # 🆕 新增
```

## ✅ 验证清单

- [ ] `aqr_weight_generator.py` 已复制
- [ ] `weight_renderer.py` 已复制
- [ ] `feature_modulator.py` 已复制
- [ ] `cmt_aqr_voxel0075_vov_1600x640_cbgs.py` 已复制
- [ ] `cmt_head.py` 已替换（原文件已备份）
- [ ] `models/utils/__init__.py` 已替换（原文件已备份）
- [ ] 标准CMT训练可正常运行（enable_aqr=False）
- [ ] AQR-CMT训练可正常运行（enable_aqr=True）
- [ ] 可以加载预训练CMT权重进行AQR微调

## 🔄 回滚方案

如果部署出现问题，可以快速回滚：

```bash
# 恢复原始cmt_head.py
cp ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/dense_heads/cmt_head_backup.py \
   ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/dense_heads/cmt_head.py

# 恢复原始__init__.py
cp ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/utils/__init_backup.py \
   ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/utils/__init__.py

# 删除AQR模块（如果需要）
rm ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/utils/aqr_weight_generator.py
rm ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/utils/weight_renderer.py
rm ${ORIGINAL_CMT}/projects/mmdet3d_plugin/models/utils/feature_modulator.py
```

## 🎯 使用说明

### 标准CMT训练（不使用AQR）
```bash
python tools/train.py projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py
```

### AQR-CMT训练（使用AQR）
```bash
# 从头训练
python tools/train.py projects/configs/fusion/cmt_aqr_voxel0075_vov_1600x640_cbgs.py

# 加载预训练CMT权重微调
python tools/train.py projects/configs/fusion/cmt_aqr_voxel0075_vov_1600x640_cbgs.py \
    --load-from work_dirs/cmt_pretrained/epoch_24.pth
```

---

**🐾 部署完成后，您的CMT项目将同时支持标准模式和AQR增强模式！**



