#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AQR权重图渲染机制集成测试脚本
测试各个模块的基本功能和集成效果
"""

import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'projects'))

def test_aqr_weight_generator():
    """测试AQR权重生成器"""
    print("🧪 Testing AQRWeightGenerator...")
    
    from mmdet3d_plugin.models.utils.aqr_weight_generator import AQRWeightGenerator
    
    # 创建测试配置
    config = dict(
        embed_dims=256,
        encoder_config=dict(
            type='TransformerLayerSequence',
            num_layers=1,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1
                ),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True)
                ),
                feedforward_channels=1024,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
            )
        ),
        window_sizes=[15, 5],
        use_type_embed=True
    )
    
    # 创建模型
    generator = AQRWeightGenerator(**config)
    
    # 创建测试数据
    batch_size = 2
    num_queries = 900
    embed_dims = 256
    total_elements = 180 * 180 + 6 * 40 * 100  # BEV + perspective
    
    query_embed = torch.randn(num_queries, batch_size, embed_dims)
    memory = torch.randn(total_elements, batch_size, embed_dims)
    pos_embed = torch.randn(total_elements, batch_size, embed_dims)
    ref_points = torch.rand(batch_size, num_queries, 3)
    
    # 模拟img_metas
    img_metas = [
        {
            'lidar2img': [np.random.randn(4, 4) for _ in range(6)],
            'img_shape': [(900, 1600, 3) for _ in range(6)]
        }
        for _ in range(batch_size)
    ]
    
    try:
        lidar_weights, camera_weights, weight_loss, projection_info = generator(
            query_embed, memory, pos_embed, ref_points, img_metas
        )
        
        print(f"   ✅ LiDAR weights shape: {lidar_weights.shape}")
        print(f"   ✅ Camera weights shape: {camera_weights.shape}")
        print(f"   ✅ Weight ranges: LiDAR [{lidar_weights.min():.3f}, {lidar_weights.max():.3f}], "
              f"Camera [{camera_weights.min():.3f}, {camera_weights.max():.3f}]")
        print(f"   ✅ Projection info keys: {list(projection_info.keys())}")
        
    except Exception as e:
        print(f"   ❌ AQRWeightGenerator test failed: {e}")
        return False
    
    return True


def test_weight_renderer():
    """测试权重图渲染器"""
    print("\n🎨 Testing WeightRenderer...")
    
    from mmdet3d_plugin.models.utils.weight_renderer import WeightRenderer
    
    # 创建渲染器
    renderer = WeightRenderer(
        render_method='gaussian',
        gaussian_sigma=2.0,
        bev_feature_shape=(180, 180),
        pers_feature_shape=(6, 40, 100)
    )
    
    # 创建测试数据
    batch_size = 2
    num_queries = 900
    
    query_weights = torch.rand(batch_size, num_queries)
    pts_bev = torch.randint(0, 180, (batch_size, num_queries, 2)).float()
    pts_pers = torch.cat([
        torch.randint(0, 6, (batch_size, num_queries, 1)).float(),
        torch.randint(0, 40, (batch_size, num_queries, 1)).float(),
        torch.randint(0, 100, (batch_size, num_queries, 1)).float()
    ], dim=-1)
    
    try:
        # 测试BEV权重渲染
        weight_map_bev = renderer.render_bev_weights(query_weights, pts_bev)
        print(f"   ✅ BEV weight map shape: {weight_map_bev.shape}")
        print(f"   ✅ BEV weight range: [{weight_map_bev.min():.3f}, {weight_map_bev.max():.3f}]")
        
        # 测试透视权重渲染
        weight_map_pers = renderer.render_perspective_weights(query_weights, pts_pers)
        print(f"   ✅ Perspective weight map shape: {weight_map_pers.shape}")
        print(f"   ✅ Perspective weight range: [{weight_map_pers.min():.3f}, {weight_map_pers.max():.3f}]")
        
    except Exception as e:
        print(f"   ❌ WeightRenderer test failed: {e}")
        return False
    
    return True


def test_feature_modulator():
    """测试特征调制器"""
    print("\n🔧 Testing FeatureModulator...")
    
    from mmdet3d_plugin.models.utils.feature_modulator import FeatureModulator
    
    # 创建调制器
    modulator = FeatureModulator(
        modulation_type='element_wise',
        residual_connection=True,
        residual_weight=0.1
    )
    
    # 创建测试数据
    batch_size = 2
    channels = 256
    
    # BEV特征测试
    bev_features = torch.randn(batch_size, channels, 180, 180)
    bev_weights = torch.rand(batch_size, 180, 180)
    
    # 透视特征测试
    views = 6
    pers_features = torch.randn(batch_size * views, channels, 40, 100)
    pers_weights = torch.rand(batch_size, views, 40, 100)
    
    try:
        # 测试BEV调制
        modulated_bev = modulator(bev_features, bev_weights, feature_type='bev')
        print(f"   ✅ BEV modulated features shape: {modulated_bev.shape}")
        
        # 计算调制效果
        diff_bev = (modulated_bev - bev_features).abs().mean()
        print(f"   ✅ BEV modulation effect (mean diff): {diff_bev:.6f}")
        
        # 测试透视调制
        modulated_pers = modulator(pers_features, pers_weights, feature_type='perspective')
        print(f"   ✅ Perspective modulated features shape: {modulated_pers.shape}")
        
        # 计算调制效果
        diff_pers = (modulated_pers - pers_features).abs().mean()
        print(f"   ✅ Perspective modulation effect (mean diff): {diff_pers:.6f}")
        
    except Exception as e:
        print(f"   ❌ FeatureModulator test failed: {e}")
        return False
    
    return True


def test_cmt_aqr_head():
    """测试CMT AQR Head集成"""
    print("\n🚀 Testing CmtAQRHead Integration...")
    
    try:
        from mmdet3d_plugin.models.dense_heads.cmt_aqr_head import CmtAQRHead
        
        # 这里只测试模块导入和基本配置
        # 完整的前向传播需要更复杂的环境设置
        print("   ✅ CmtAQRHead imported successfully")
        
        # 测试配置生成
        from mmdet3d_plugin.models.dense_heads.cmt_aqr_head import get_cmt_aqr_config
        config = get_cmt_aqr_config()
        print("   ✅ Configuration generated successfully")
        print(f"   ✅ Config type: {config['type']}")
        print(f"   ✅ AQR enabled: {config['enable_aqr']}")
        
    except Exception as e:
        print(f"   ❌ CmtAQRHead test failed: {e}")
        return False
    
    return True


def main():
    """主测试函数"""
    print("🎯 AQR权重图渲染机制集成测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行各个测试
    test_results.append(test_aqr_weight_generator())
    test_results.append(test_weight_renderer())
    test_results.append(test_feature_modulator())
    test_results.append(test_cmt_aqr_head())
    
    # 总结测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    test_names = [
        "AQRWeightGenerator",
        "WeightRenderer", 
        "FeatureModulator",
        "CmtAQRHead"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {i+1}. {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎉 测试完成: {passed}/{len(test_results)} 项测试通过")
    
    if passed == len(test_results):
        print("🎊 所有测试通过！AQR权重图渲染机制已成功集成到CMT框架中。")
        print("\n📝 下一步:")
        print("   1. 使用配置文件 'cmt_aqr_voxel0075_vov_1600x640_cbgs.py' 开始训练")
        print("   2. 根据需要调整权重渲染和调制参数")
        print("   3. 监控训练过程中的权重分布和调制效果")
    else:
        print("⚠️  部分测试失败，请检查相关模块的实现。")
    
    return passed == len(test_results)


if __name__ == "__main__":
    main()
