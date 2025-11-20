#!/usr/bin/env python
"""
测试配置文件是否正确加载AQR设置
"""
import sys
sys.path.insert(0, '.')

from mmcv import Config

# 加载配置
config_file = 'projects/configs/fusion/cmt_aqr_voxel0100_r50_800x320_cbgs.py'
print(f"📋 Loading config from: {config_file}")

cfg = Config.fromfile(config_file)

# 🔥 打印原始配置中pts_bbox_head的所有键
print(f"\n🔍 pts_bbox_head keys: {list(cfg.model.pts_bbox_head.keys())}")
print(f"   检查是否有 enable_aqr: {'enable_aqr' in cfg.model.pts_bbox_head}")

# 检查model配置
print(f"\n{'='*70}")
print(f"🔍 Model Configuration Check:")
print(f"{'='*70}")

if hasattr(cfg, 'model'):
    print(f"✅ cfg.model exists")
    
    if 'pts_bbox_head' in cfg.model:
        head_cfg = cfg.model.pts_bbox_head
        print(f"✅ cfg.model.pts_bbox_head exists")
        print(f"   type = {head_cfg.get('type', 'NOT FOUND')}")
        print(f"   enable_aqr = {head_cfg.get('enable_aqr', 'NOT FOUND')}")
        print(f"   debug_mode = {head_cfg.get('debug_mode', 'NOT FOUND')}")
        
        if 'aqr_config' in head_cfg:
            if head_cfg.aqr_config is not None:
                print(f"✅ aqr_config exists")
                print(f"   Keys: {list(head_cfg.aqr_config.keys())}")
            else:
                print(f"⚠️  aqr_config is None (not configured)")
        else:
            print(f"❌ aqr_config NOT FOUND")
            
        if 'attention_bias_config' in head_cfg:
            if head_cfg.attention_bias_config is not None:
                print(f"✅ attention_bias_config exists")
                print(f"   Keys: {list(head_cfg.attention_bias_config.keys())}")
                print(f"   debug_print = {head_cfg.attention_bias_config.get('debug_print', 'NOT FOUND')}")
                print(f"   print_interval = {head_cfg.attention_bias_config.get('print_interval', 'NOT FOUND')}")
            else:
                print(f"⚠️  attention_bias_config is None (not configured)")
        else:
            print(f"❌ attention_bias_config NOT FOUND")
    else:
        print(f"❌ cfg.model.pts_bbox_head NOT FOUND")
        print(f"   Available keys: {list(cfg.model.keys())}")
else:
    print(f"❌ cfg.model NOT FOUND")

print(f"{'='*70}\n")

# 尝试构建模型
print(f"🔨 Trying to build model...")
try:
    from mmdet3d.models import build_detector
    model = build_detector(cfg.model)
    print(f"✅ Model built successfully!")
    print(f"   Model type: {type(model).__name__}")
    
    if hasattr(model, 'pts_bbox_head'):
        head = model.pts_bbox_head
        print(f"   Head type: {type(head).__name__}")
        print(f"   Head.enable_aqr: {getattr(head, 'enable_aqr', 'NOT FOUND')}")
        print(f"   Head.debug_mode: {getattr(head, 'debug_mode', 'NOT FOUND')}")
    else:
        print(f"   ❌ Model has no pts_bbox_head attribute")
        
except Exception as e:
    print(f"❌ Model build failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🎯 Test complete!")

