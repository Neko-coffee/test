#!/usr/bin/env python3
"""
🔬 全面诊断 bias_scale 不学习的问题
"""
import sys
sys.path.insert(0, '.')

import torch
from mmcv import Config
from mmdet3d.models import build_model

# 🔥 导入自定义模块以注册所有组件
import projects.mmdet3d_plugin  # noqa: F401

def diagnose_bias_scale():
    """全面诊断"""
    
    print("="*80)
    print("🔬 Bias Scale 学习问题诊断")
    print("="*80)
    
    # Step 1: 加载配置
    print("\n📋 Step 1: 加载配置文件")
    print("-"*80)
    config_file = 'projects/configs/fusion/cmt_aqr_voxel0075_vov_1600x640_cbgs.py'
    cfg = Config.fromfile(config_file)
    
    # 检查配置中的 optimizer
    if hasattr(cfg, 'optimizer') and 'paramwise_cfg' in cfg.optimizer:
        custom_keys = cfg.optimizer['paramwise_cfg']['custom_keys']
        print("✅ 配置文件中的 optimizer.paramwise_cfg.custom_keys:")
        for key, value in custom_keys.items():
            if 'bias' in key.lower():
                print(f"   🔥 {key}: {value}")
    else:
        print("❌ 配置文件中没有找到 paramwise_cfg!")
    
    # Step 2: 构建模型
    print("\n🔨 Step 2: 构建模型")
    print("-"*80)
    try:
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        print("✅ 模型构建成功")
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        return
    
    # Step 3: 检查 bias_scale 是否存在
    print("\n🔍 Step 3: 查找 bias_scale 参数")
    print("-"*80)
    
    bias_scale_found = False
    bias_scale_param = None
    bias_scale_name = None
    
    for name, param in model.named_parameters():
        if 'bias_scale' in name:
            bias_scale_found = True
            bias_scale_param = param
            bias_scale_name = name
            
            print(f"✅ 找到 bias_scale 参数!")
            print(f"   完整参数名: {name}")
            print(f"   当前值: {param.item():.6f}")
            print(f"   requires_grad: {param.requires_grad}")
            print(f"   is_leaf: {param.is_leaf}")
            print(f"   shape: {param.shape}")
            print(f"   dtype: {param.dtype}")
            print(f"   device: {param.device}")
    
    if not bias_scale_found:
        print("❌ 没有找到 bias_scale 参数!")
        print("\n可能的原因:")
        print("1. enable_aqr=False")
        print("2. learnable_scale=False")
        print("3. attention_bias_generator 没有正确初始化")
        return
    
    # Step 4: 检查优化器配置匹配
    print("\n🔍 Step 4: 检查优化器配置匹配")
    print("-"*80)
    
    # 列出所有可能的配置键
    possible_keys = [
        'bias_scale',
        'attention_bias_generator.bias_scale',
        'pts_bbox_head.attention_bias_generator.bias_scale',
    ]
    
    print(f"实际参数名: {bias_scale_name}")
    print(f"\n检查配置文件中的键:")
    
    for key in possible_keys:
        is_in_config = key in custom_keys
        would_match = key in bias_scale_name
        
        status = "✅" if is_in_config else "❌"
        match_status = "✅" if would_match else "❌"
        
        print(f"  {status} '{key}'")
        print(f"     - 在配置中: {is_in_config}")
        print(f"     - 会匹配参数名: {would_match}")
        
        if is_in_config:
            print(f"     - lr_mult: {custom_keys[key].get('lr_mult', 'N/A')}")
    
    # Step 5: 模拟优化器设置
    print("\n🧪 Step 5: 模拟优化器参数组构建")
    print("-"*80)
    
    # 检查哪个配置会匹配
    matched_config = None
    for key in possible_keys:
        if key in custom_keys and key in bias_scale_name:
            matched_config = key
            break
    
    if matched_config:
        print(f"✅ 参数 '{bias_scale_name}' 会被配置键 '{matched_config}' 匹配")
        print(f"   lr_mult: {custom_keys[matched_config]['lr_mult']}")
    else:
        print(f"❌ 参数 '{bias_scale_name}' 不会被任何配置键匹配!")
        print(f"\n🔧 建议修复:")
        print(f"   在配置文件中添加: '{bias_scale_name}': dict(lr_mult=1.0)")
    
    # Step 6: 检查所有 attention_bias 相关参数
    print("\n📊 Step 6: 所有 attention_bias 相关参数")
    print("-"*80)
    
    count = 0
    for name, param in model.named_parameters():
        if 'attention_bias' in name:
            count += 1
            print(f"{count}. {name}")
            print(f"   - requires_grad: {param.requires_grad}")
            print(f"   - shape: {param.shape}")
    
    if count == 0:
        print("❌ 没有找到任何 attention_bias 相关参数!")
    
    # Step 7: 测试梯度计算
    print("\n🧪 Step 7: 测试梯度计算")
    print("-"*80)
    
    if bias_scale_param is not None and bias_scale_param.requires_grad:
        # 创建一个简单的损失函数
        dummy_loss = bias_scale_param * 2.0
        dummy_loss.backward()
        
        if bias_scale_param.grad is not None:
            print(f"✅ 梯度计算成功!")
            print(f"   grad: {bias_scale_param.grad.item():.6f}")
        else:
            print(f"❌ 梯度计算失败! grad is None")
    else:
        print(f"❌ 无法测试梯度 (requires_grad={bias_scale_param.requires_grad if bias_scale_param else 'N/A'})")
    
    # Step 8: 生成修复建议
    print("\n" + "="*80)
    print("🔧 诊断总结和修复建议")
    print("="*80)
    
    if not bias_scale_found:
        print("❌ 问题: bias_scale 参数不存在")
        print("\n修复步骤:")
        print("1. 检查配置文件中 enable_aqr=True")
        print("2. 检查配置文件中 learnable_scale=True")
        print("3. 检查 attention_bias_config 是否正确配置")
    elif not matched_config:
        print("❌ 问题: bias_scale 参数存在但不会被优化器匹配")
        print(f"\n实际参数名: {bias_scale_name}")
        print(f"\n修复步骤:")
        print(f"1. 在配置文件的 optimizer.paramwise_cfg.custom_keys 中添加:")
        print(f"   '{bias_scale_name}': dict(lr_mult=1.0),")
        print(f"\n2. 或者检查当前配置是否有拼写错误")
    else:
        print("✅ 配置看起来正确!")
        print(f"   参数名: {bias_scale_name}")
        print(f"   匹配的配置键: {matched_config}")
        print(f"   lr_mult: {custom_keys[matched_config]['lr_mult']}")
        print("\n如果训练时仍然不更新，可能的原因:")
        print("1. 服务器代码未同步 - 请检查服务器上的文件")
        print("2. 使用了错误的配置文件")
        print("3. checkpoint中加载了旧的值")
        print("4. 学习率过小或梯度过小")
    
    # Step 9: 生成文件同步检查脚本
    print("\n" + "="*80)
    print("📁 生成文件同步检查脚本")
    print("="*80)
    
    sync_check_script = """#!/bin/bash
# 检查关键文件的最后修改时间

echo "==================================================================="
echo "🔍 检查关键文件是否已同步"
echo "==================================================================="

files=(
    "projects/configs/fusion/cmt_aqr_voxel0075_vov_1600x640_cbgs.py"
    "projects/mmdet3d_plugin/models/utils/attention_bias_generator.py"
    "projects/mmdet3d_plugin/models/dense_heads/cmt_head.py"
)

for file in "${files[@]}"; do
    echo ""
    echo "📄 文件: $file"
    if [ -f "$file" ]; then
        echo "   ✅ 存在"
        echo "   最后修改: $(stat -c '%y' "$file" 2>/dev/null || stat -f '%Sm' "$file")"
        echo "   大小: $(stat -c '%s' "$file" 2>/dev/null || stat -f '%z' "$file") bytes"
        
        # 检查关键行
        if [[ "$file" == *"cmt_aqr_voxel0075_vov_1600x640_cbgs.py" ]]; then
            echo "   检查第35行 (bias_scale 配置):"
            sed -n '35p' "$file"
        fi
        
        if [[ "$file" == *"attention_bias_generator.py" ]]; then
            echo "   检查第88行 (nn.Parameter 创建):"
            sed -n '88p' "$file"
        fi
    else
        echo "   ❌ 文件不存在!"
    fi
done

echo ""
echo "==================================================================="
"""
    
    with open('CMT-master/check_file_sync.sh', 'w') as f:
        f.write(sync_check_script)
    
    print("✅ 已生成 check_file_sync.sh")
    print("   在服务器上运行: bash check_file_sync.sh")
    
    print("\n🎯 诊断完成!")

if __name__ == '__main__':
    diagnose_bias_scale()

