#!/usr/bin/env python3
"""
AQR权重图可视化工具 🐾
用于可视化AQR生成的权重分配，验证权重是否集中在目标位置

使用方法:
1. 训练时启用debug_mode和可视化:
   python tools/train.py configs/fusion/cmt_aqr_config.py --debug-aqr

2. 单独可视化已保存的权重:
   python tools/visualize_aqr_weights.py --config configs/fusion/cmt_aqr_config.py \
       --checkpoint work_dirs/latest.pth --save-dir viz_output/
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2

from mmdet3d.apis import init_model, inference_detector
from mmcv import Config
from mmdet3d.datasets import build_dataloader, build_dataset


def visualize_weight_map_with_boxes(weight_map, gt_boxes_2d=None, pred_boxes_2d=None, 
                                   save_path='weight_viz.png', title='AQR Weight Map'):
    """
    可视化权重图，并叠加GT和预测框
    
    Args:
        weight_map: [H, W] 权重图
        gt_boxes_2d: List of [x, y, w, h] GT框在特征图上的坐标
        pred_boxes_2d: List of [x, y, w, h] 预测框在特征图上的坐标
        save_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(12, 10))
    
    # 创建自定义颜色映射：蓝(低权重) -> 绿 -> 黄 -> 红(高权重)
    colors = ['#000033', '#0000FF', '#00FF00', '#FFFF00', '#FF0000']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('aqr_weights', colors, N=n_bins)
    
    # 绘制权重图
    im = plt.imshow(weight_map, cmap=cmap, interpolation='bilinear', aspect='auto')
    plt.colorbar(im, label='Weight Value', fraction=0.046, pad=0.04)
    
    # 叠加GT框（绿色）
    if gt_boxes_2d is not None and len(gt_boxes_2d) > 0:
        for box in gt_boxes_2d:
            x, y, w, h = box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                     edgecolor='lime', facecolor='none', 
                                     label='GT Box')
            plt.gca().add_patch(rect)
    
    # 叠加预测框（黄色）
    if pred_boxes_2d is not None and len(pred_boxes_2d) > 0:
        for box in pred_boxes_2d:
            x, y, w, h = box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                     edgecolor='yellow', facecolor='none', 
                                     linestyle='--', label='Pred Box')
            plt.gca().add_patch(rect)
    
    # 移除重复的图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Feature Width', fontsize=12)
    plt.ylabel('Feature Height', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def project_3d_boxes_to_bev(boxes_3d, pc_range=[-54, -54, -5, 54, 54, 3], 
                           feature_size=(180, 180)):
    """
    将3D框投影到BEV特征图坐标
    
    Args:
        boxes_3d: [N, 7] (x, y, z, w, l, h, yaw) LiDAR坐标系
        pc_range: 点云范围
        feature_size: BEV特征图尺寸
        
    Returns:
        boxes_2d: List of [x, y, w, h] BEV特征图坐标
    """
    if boxes_3d is None or len(boxes_3d) == 0:
        return []
    
    boxes_2d = []
    pc_min = np.array([pc_range[0], pc_range[1]])
    pc_max = np.array([pc_range[3], pc_range[4]])
    pc_size = pc_max - pc_min
    
    for box in boxes_3d:
        x, y, w, l = box[0], box[1], box[3], box[4]  # 注意：w和l的顺序
        
        # 归一化到[0, 1]
        x_norm = (x - pc_min[0]) / pc_size[0]
        y_norm = (y - pc_min[1]) / pc_size[1]
        w_norm = w / pc_size[0]
        l_norm = l / pc_size[1]
        
        # 转换到特征图坐标
        feat_x = x_norm * feature_size[1]
        feat_y = y_norm * feature_size[0]
        feat_w = w_norm * feature_size[1]
        feat_l = l_norm * feature_size[0]
        
        # BEV框：[x-l/2, y-w/2, l, w] (注意坐标系转换)
        boxes_2d.append([
            feat_y - feat_w/2,  # y坐标
            feat_x - feat_l/2,  # x坐标
            feat_w,             # 宽度
            feat_l              # 长度
        ])
    
    return boxes_2d


def visualize_perspective_weights(weight_maps_pers, img_metas, save_dir='viz_output/'):
    """
    可视化透视视角的权重图（6个相机视角）
    
    Args:
        weight_maps_pers: [bs, 6, 40, 100] 或 [6, 40, 100] 透视权重图
        img_metas: 图像元数据
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if weight_maps_pers.dim() == 4:
        weight_maps_pers = weight_maps_pers[0]  # 取第一个batch
    
    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('AQR Perspective Weight Maps (All Camera Views)', 
                fontsize=16, fontweight='bold')
    
    for view_idx in range(6):
        ax = axes[view_idx // 3, view_idx % 3]
        weight_map = weight_maps_pers[view_idx].cpu().numpy()
        
        # 创建颜色映射
        colors = ['#000033', '#0000FF', '#00FF00', '#FFFF00', '#FF0000']
        cmap = LinearSegmentedColormap.from_list('aqr_weights', colors, N=100)
        
        im = ax.imshow(weight_map, cmap=cmap, interpolation='bilinear', aspect='auto')
        ax.set_title(f'{camera_names[view_idx]}\nWeight Range: [{weight_map.min():.3f}, {weight_map.max():.3f}]')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'perspective_weights_all_views.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def analyze_weight_target_correlation(weight_map, gt_boxes_2d, feature_size=(180, 180)):
    """
    分析权重与目标位置的相关性
    
    Args:
        weight_map: [H, W] 权重图
        gt_boxes_2d: List of [x, y, w, h] GT框坐标
        
    Returns:
        correlation_stats: Dict 包含统计信息
    """
    stats = {
        'total_weight': weight_map.sum(),
        'mean_weight': weight_map.mean(),
        'max_weight': weight_map.max(),
        'weight_in_boxes': 0.0,
        'weight_outside_boxes': 0.0,
        'boxes_count': len(gt_boxes_2d),
        'coverage_ratio': 0.0
    }
    
    if len(gt_boxes_2d) == 0:
        stats['weight_outside_boxes'] = stats['total_weight']
        return stats
    
    # 创建目标区域mask
    mask = np.zeros(feature_size, dtype=bool)
    for box in gt_boxes_2d:
        x, y, w, h = [int(v) for v in box]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(feature_size[1], x + w), min(feature_size[0], y + h)
        mask[y1:y2, x1:x2] = True
    
    # 计算框内和框外权重
    stats['weight_in_boxes'] = weight_map[mask].sum()
    stats['weight_outside_boxes'] = weight_map[~mask].sum()
    
    if stats['total_weight'] > 0:
        stats['coverage_ratio'] = stats['weight_in_boxes'] / stats['total_weight']
    
    return stats


def visualize_weight_statistics(stats_list, save_path='weight_stats.png'):
    """
    可视化权重统计信息
    
    Args:
        stats_list: List of dicts 每个样本的统计信息
        save_path: 保存路径
    """
    if len(stats_list) == 0:
        return
    
    coverage_ratios = [s['coverage_ratio'] for s in stats_list]
    boxes_counts = [s['boxes_count'] for s in stats_list]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 权重覆盖率分布
    ax1.hist(coverage_ratios, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(coverage_ratios), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(coverage_ratios):.2%}')
    ax1.set_xlabel('Weight Coverage Ratio (In Boxes / Total)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Weight Coverage on GT Objects', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 目标数量 vs 覆盖率散点图
    ax2.scatter(boxes_counts, coverage_ratios, alpha=0.6, s=50, c='green', edgecolors='black')
    ax2.set_xlabel('Number of GT Objects', fontsize=12)
    ax2.set_ylabel('Weight Coverage Ratio', fontsize=12)
    ax2.set_title('Objects Count vs Weight Coverage', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='AQR权重可视化工具')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', help='模型checkpoint路径')
    parser.add_argument('--save-dir', default='aqr_viz_output/', help='可视化结果保存目录')
    parser.add_argument('--num-samples', type=int, default=10, help='可视化样本数量')
    parser.add_argument('--data-split', default='val', choices=['train', 'val', 'test'], 
                       help='数据集分割')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载配置和模型
    cfg = Config.fromfile(args.config)
    
    # 确保启用AQR和debug模式
    if hasattr(cfg.model, 'pts_bbox_head'):
        cfg.model.pts_bbox_head.enable_aqr = True
        cfg.model.pts_bbox_head.debug_mode = True
    
    # 构建数据集
    if args.data_split == 'val':
        dataset = build_dataset(cfg.data.val)
    elif args.data_split == 'train':
        dataset = build_dataset(cfg.data.train.dataset)
    else:
        dataset = build_dataset(cfg.data.test)
    
    print(f"📊 Dataset: {len(dataset)} samples")
    print(f"🎯 Will visualize {min(args.num_samples, len(dataset))} samples")
    
    # 初始化模型（如果提供了checkpoint）
    if args.checkpoint:
        model = init_model(cfg, args.checkpoint, device='cuda:0')
        print(f"✅ Model loaded from {args.checkpoint}")
    
    # 可视化样本
    stats_list = []
    
    for idx in range(min(args.num_samples, len(dataset))):
        print(f"\n🔍 Processing sample {idx+1}/{min(args.num_samples, len(dataset))}...")
        
        data = dataset[idx]
        img_metas = data['img_metas'].data
        gt_bboxes_3d = data.get('gt_bboxes_3d', None)
        
        # 这里需要从模型中提取权重图
        # 实际使用时，需要在cmt_head.py中保存权重图到文件或返回
        print(f"⚠️  Note: 需要在训练时启用debug_mode并保存权重图")
        print(f"   可以在cmt_head.py的_apply_aqr_modulation中添加:")
        print(f"   torch.save({{'weight_map_bev': weight_map_bev, 'weight_map_pers': weight_map_pers}}, 'debug_weights.pth')")
    
    print(f"\n✅ Visualization completed! Results saved to {args.save_dir}")


if __name__ == '__main__':
    main()



