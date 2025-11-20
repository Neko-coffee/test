#!/usr/bin/env python
"""
注意力可视化工具
用于对比有AQR和无AQR时的注意力权重分布

使用方法:
python tools/visualize_attention.py \
    --config-aqr configs/fusion/cmt_aqr_voxel0100_r50_800x320_cbgs.py \
    --config-baseline configs/fusion/cmt_baseline_voxel0100_r50_800x320_cbgs.py \
    --checkpoint-aqr work_dirs/cmt_aqr/latest.pth \
    --checkpoint-baseline work_dirs/cmt_baseline/latest.pth \
    --sample-idx 0 \
    --query-idx 100 \
    --save-dir attention_vis
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector

def enable_attention_saving(model):
    """启用模型中所有attention模块的权重保存"""
    for name, module in model.named_modules():
        if hasattr(module, 'attn') and hasattr(module.attn, 'forward'):
            module.save_attn_weights = True
            print(f"✅ Enabled attention saving for: {name}")

def extract_attention_weights(model):
    """提取模型中保存的注意力权重"""
    attn_weights_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'last_attn_weights'):
            attn_weights_dict[name] = {
                'weights': module.last_attn_weights.detach().cpu(),
                'bias': getattr(module, 'last_attention_bias', None)
            }
    return attn_weights_dict

def visualize_attention_comparison(
    attn_aqr, attn_baseline, 
    query_idx=0, 
    layer_name='decoder.layers.5',  # 最后一层
    save_path='attention_comparison.png'
):
    """
    对比可视化有AQR和无AQR的注意力权重
    
    Args:
        attn_aqr: AQR模型的注意力权重字典
        attn_baseline: Baseline模型的注意力权重字典
        query_idx: 要可视化的query索引
        layer_name: 要可视化的层名
        save_path: 保存路径
    """
    # 提取特定层的注意力权重
    if layer_name not in attn_aqr or layer_name not in attn_baseline:
        print(f"⚠️ Layer {layer_name} not found in attention weights")
        available_layers = list(attn_aqr.keys())
        print(f"Available layers: {available_layers}")
        if available_layers:
            layer_name = available_layers[-1]
            print(f"Using layer: {layer_name}")
        else:
            return
    
    # AQR模型的注意力权重
    weights_aqr = attn_aqr[layer_name]['weights']  # [bs*num_heads, num_queries, num_features]
    bias_aqr = attn_aqr[layer_name]['bias']
    
    # Baseline模型的注意力权重
    weights_baseline = attn_baseline[layer_name]['weights']
    
    # 取第一个样本，所有头的平均
    num_heads = 8  # 根据实际配置调整
    bs = weights_aqr.shape[0] // num_heads
    
    # 重塑并平均所有头
    weights_aqr = weights_aqr.view(bs, num_heads, -1, weights_aqr.shape[-1])
    weights_aqr = weights_aqr[0].mean(dim=0)  # [num_queries, num_features]
    
    weights_baseline = weights_baseline.view(bs, num_heads, -1, weights_baseline.shape[-1])
    weights_baseline = weights_baseline[0].mean(dim=0)  # [num_queries, num_features]
    
    # 提取特定query的注意力分布
    query_attn_aqr = weights_aqr[query_idx].numpy()  # [num_features]
    query_attn_baseline = weights_baseline[query_idx].numpy()
    
    # 分离BEV和Camera部分（假设BEV=180*180=32400, Camera=6*40*100=24000）
    bev_size = 180 * 180
    
    bev_attn_aqr = query_attn_aqr[:bev_size].reshape(180, 180)
    cam_attn_aqr = query_attn_aqr[bev_size:].reshape(-1)  # 展平camera部分
    
    bev_attn_baseline = query_attn_baseline[:bev_size].reshape(180, 180)
    cam_attn_baseline = query_attn_baseline[bev_size:].reshape(-1)
    
    # 创建可视化
    fig = plt.figure(figsize=(20, 10))
    
    # Row 1: AQR模型的注意力
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(bev_attn_aqr, cmap='hot', interpolation='bilinear')
    ax1.set_title(f'AQR - BEV Attention (Query {query_idx})', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(cam_attn_aqr, 'r-', linewidth=1)
    ax2.set_title(f'AQR - Camera Attention (Query {query_idx})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Attention Weight')
    ax2.grid(True, alpha=0.3)
    
    # Row 2: Baseline模型的注意力
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(bev_attn_baseline, cmap='hot', interpolation='bilinear')
    ax4.set_title(f'Baseline - BEV Attention (Query {query_idx})', fontsize=14, fontweight='bold')
    plt.colorbar(im4, ax=ax4)
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(cam_attn_baseline, 'b-', linewidth=1)
    ax5.set_title(f'Baseline - Camera Attention (Query {query_idx})', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Feature Index')
    ax5.set_ylabel('Attention Weight')
    ax5.grid(True, alpha=0.3)
    
    # Row 1, Col 3: 差异图（BEV）
    ax3 = plt.subplot(2, 3, 3)
    diff_bev = bev_attn_aqr - bev_attn_baseline
    im3 = ax3.imshow(diff_bev, cmap='RdBu_r', interpolation='bilinear', vmin=-diff_bev.max(), vmax=diff_bev.max())
    ax3.set_title(f'Difference (AQR - Baseline)\nBEV Attention', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Attention Difference')
    
    # Row 2, Col 3: 差异图（Camera）
    ax6 = plt.subplot(2, 3, 6)
    diff_cam = cam_attn_aqr - cam_attn_baseline
    ax6.plot(diff_cam, 'g-', linewidth=1)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.set_title(f'Difference (AQR - Baseline)\nCamera Attention', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Feature Index')
    ax6.set_ylabel('Attention Difference')
    ax6.grid(True, alpha=0.3)
    
    # 统计信息
    stats_text = f"""
    Query {query_idx} Statistics:
    
    BEV Attention:
    - AQR max: {bev_attn_aqr.max():.4f}
    - Baseline max: {bev_attn_baseline.max():.4f}
    - Diff mean: {diff_bev.mean():.4f}
    
    Camera Attention:
    - AQR max: {cam_attn_aqr.max():.4f}
    - Baseline max: {cam_attn_baseline.max():.4f}
    - Diff mean: {diff_cam.mean():.4f}
    """
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, family='monospace', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved attention visualization to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize Attention Weights Comparison')
    parser.add_argument('--config-aqr', required=True, help='AQR model config file')
    parser.add_argument('--config-baseline', required=True, help='Baseline model config file')
    parser.add_argument('--checkpoint-aqr', required=True, help='AQR model checkpoint')
    parser.add_argument('--checkpoint-baseline', required=True, help='Baseline model checkpoint')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--query-idx', type=int, default=100, help='Query index to visualize')
    parser.add_argument('--save-dir', default='attention_vis', help='Directory to save visualizations')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载配置
    print("📋 Loading configurations...")
    cfg_aqr = Config.fromfile(args.config_aqr)
    cfg_baseline = Config.fromfile(args.config_baseline)
    
    # 构建数据集
    print("📊 Building dataset...")
    dataset = build_dataset(cfg_aqr.data.val)
    data = dataset[args.sample_idx]
    
    # 构建模型
    print("🔨 Building AQR model...")
    model_aqr = build_detector(cfg_aqr.model)
    checkpoint_aqr = torch.load(args.checkpoint_aqr, map_location='cpu')
    model_aqr.load_state_dict(checkpoint_aqr['state_dict'])
    model_aqr.eval()
    enable_attention_saving(model_aqr)
    
    print("🔨 Building Baseline model...")
    model_baseline = build_detector(cfg_baseline.model)
    checkpoint_baseline = torch.load(args.checkpoint_baseline, map_location='cpu')
    model_baseline.load_state_dict(checkpoint_baseline['state_dict'])
    model_baseline.eval()
    enable_attention_saving(model_baseline)
    
    # 前向传播
    print("🚀 Running forward pass with AQR...")
    with torch.no_grad():
        _ = model_aqr(**data)
    attn_aqr = extract_attention_weights(model_aqr)
    
    print("🚀 Running forward pass with Baseline...")
    with torch.no_grad():
        _ = model_baseline(**data)
    attn_baseline = extract_attention_weights(model_baseline)
    
    # 可视化
    print("🎨 Visualizing attention weights...")
    save_path = os.path.join(args.save_dir, f'attention_comparison_query{args.query_idx}.png')
    visualize_attention_comparison(
        attn_aqr, attn_baseline,
        query_idx=args.query_idx,
        save_path=save_path
    )
    
    print("✅ Done!")

if __name__ == '__main__':
    main()

