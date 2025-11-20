#!/usr/bin/env python
"""
测试 Attention Bias 集成
验证所有组件是否正确连接
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_attention_bias_generator():
    """测试AttentionBiasGenerator"""
    from projects.mmdet3d_plugin.models.utils.attention_bias_generator import AttentionBiasGenerator
    
    print("🔥 测试 AttentionBiasGenerator...")
    
    # 创建生成器
    generator = AttentionBiasGenerator(
        bev_feature_shape=(128, 128),
        pers_feature_shape=(6, 20, 50),
        window_size=8,
        bias_scale=1.0,
        use_local_bias=True,
        fp16=False
    )
    
    # 模拟输入
    bs, num_queries = 2, 900
    lidar_weights = torch.rand(bs, num_queries)
    camera_weights = torch.rand(bs, num_queries)
    pts_bev = torch.randint(0, 128*128, (bs, num_queries))
    pts_pers = torch.randint(0, 6*20*50, (bs, num_queries))
    
    # 前向传播
    attention_bias = generator(lidar_weights, camera_weights, pts_bev, pts_pers, img_metas=None)
    
    # 验证输出
    expected_shape = (bs, num_queries, 128*128 + 6*20*50)
    assert attention_bias.shape == expected_shape, f"形状错误：{attention_bias.shape} vs {expected_shape}"
    
    print(f"   ✅ 输出形状正确: {attention_bias.shape}")
    print(f"   ✅ Bias范围: [{attention_bias.min():.4f}, {attention_bias.max():.4f}]")
    print(f"   ✅ Bias均值: {attention_bias.mean():.4f}")
    
    return True


def test_petr_multihead_attention():
    """测试PETRMultiheadAttention支持attention_bias"""
    from projects.mmdet3d_plugin.models.utils.petr_transformer import PETRMultiheadAttention
    
    print("\n🔥 测试 PETRMultiheadAttention...")
    
    # 创建注意力模块
    attn = PETRMultiheadAttention(
        embed_dims=256,
        num_heads=8,
        dropout=0.1
    )
    
    # 模拟输入
    num_queries, bs, embed_dims = 900, 2, 256
    num_features = 128*128 + 6*20*50
    
    query = torch.randn(num_queries, bs, embed_dims)
    key = torch.randn(num_features, bs, embed_dims)
    value = torch.randn(num_features, bs, embed_dims)
    
    # 🔥 测试：不使用attention_bias
    out1 = attn(query, key=key, value=value, attention_bias=None)
    print(f"   ✅ 不使用bias输出形状: {out1.shape}")
    
    # 🔥 测试：使用attention_bias
    attention_bias = torch.randn(num_queries, bs, num_features)
    out2 = attn(query, key=key, value=value, attention_bias=attention_bias)
    print(f"   ✅ 使用bias输出形状: {out2.shape}")
    
    # 验证输出不同
    assert not torch.equal(out1, out2), "使用bias后输出应该不同"
    print(f"   ✅ Attention bias生效（输出发生变化）")
    
    return True


def test_cmt_transformer():
    """测试CmtTransformer支持attention_bias"""
    from projects.mmdet3d_plugin.models.utils.cmt_transformer import CmtTransformer
    from projects.mmdet3d_plugin.models.utils.petr_transformer import PETRTransformerDecoder
    
    print("\n🔥 测试 CmtTransformer...")
    
    # 创建Transformer
    decoder_config = dict(
        type='PETRTransformerDecoder',
        return_intermediate=True,
        num_layers=2,
        transformerlayers=dict(
            type='PETRTransformerDecoderLayer',
            attn_cfgs=[
                dict(type='MultiheadAttention', embed_dims=256, num_heads=8),
                dict(type='MultiheadAttention', embed_dims=256, num_heads=8)
            ],
            ffn_cfgs=dict(type='FFN', embed_dims=256, feedforward_channels=1024),
            feedforward_channels=1024,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        )
    )
    
    transformer = CmtTransformer(decoder=decoder_config)
    
    # 模拟输入
    bs, c, h, w = 2, 256, 128, 128
    bev_feat = torch.randn(bs, c, h, w)
    
    bs_cam, c_cam, h_cam, w_cam = bs * 6, 256, 20, 50
    cam_feat = torch.randn(bs_cam, c_cam, h_cam, w_cam)
    
    num_queries = 900
    query_embed = torch.randn(bs, num_queries, 256)
    
    bev_pos = torch.randn(h * w, bs, 256)
    rv_pos = torch.randn(6 * h_cam * w_cam, bs, 256)
    
    # 🔥 测试：不使用attention_bias
    out1, _ = transformer(bev_feat, cam_feat, query_embed, bev_pos, rv_pos, attention_bias=None)
    print(f"   ✅ 不使用bias输出形状: {out1.shape}")
    
    # 🔥 测试：使用attention_bias
    attention_bias = torch.randn(bs, num_queries, h*w + 6*h_cam*w_cam)
    out2, _ = transformer(bev_feat, cam_feat, query_embed, bev_pos, rv_pos, attention_bias=attention_bias)
    print(f"   ✅ 使用bias输出形状: {out2.shape}")
    
    # 验证输出不同
    assert not torch.equal(out1, out2), "使用bias后输出应该不同"
    print(f"   ✅ Attention bias在Transformer中生效")
    
    return True


def test_integration():
    """集成测试"""
    print("\n" + "="*60)
    print("🚀 开始 Attention Bias 集成测试")
    print("="*60 + "\n")
    
    try:
        # 测试各个组件
        test_attention_bias_generator()
        test_petr_multihead_attention()
        test_cmt_transformer()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！Attention Bias集成成功！")
        print("="*60)
        print("\n📝 后续步骤：")
        print("   1. 在配置文件中启用 enable_aqr=True")
        print("   2. 添加 attention_bias_config 配置")
        print("   3. 运行训练脚本验证端到端流程")
        print("   4. 对比旧方案（特征调制）和新方案（Attention Bias）的性能")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_integration()
    sys.exit(0 if success else 1)

