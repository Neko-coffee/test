#!/usr/bin/env python3
"""
AQR权重图快速可视化脚本 🐾

使用方法:
1. 训练时启用debug_mode:
   在配置文件中设置 debug_mode=True

2. 运行此脚本可视化保存的权重:
   python tools/plot_aqr_weights.py --weight-file aqr_debug_weights/weights_iter_100.pth
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


def project_3d_boxes_to_bev(boxes_3d, pc_range=[-54, -54, -5, 54, 54, 3], 
                           feature_size=(180, 180)):
    """将3D框投影到BEV特征图坐标"""
    if boxes_3d is None or len(boxes_3d) == 0:
        return []
    
    if hasattr(boxes_3d, 'tensor'):
        boxes_3d = boxes_3d.tensor.cpu().numpy()
    elif torch.is_tensor(boxes_3d):
        boxes_3d = boxes_3d.cpu().numpy()
    
    boxes_2d = []
    pc_min = np.array([pc_range[0], pc_range[1]])
    pc_max = np.array([pc_range[3], pc_range[4]])
    pc_size = pc_max - pc_min
    
    for box in boxes_3d:
        x, y = box[0], box[1]
        if len(box) >= 5:
            w, l = box[3], box[4]  # w=宽度, l=长度
        else:
            w, l = 4.0, 2.0  # 默认尺寸
        
        # 归一化到[0, 1]
        x_norm = (x - pc_min[0]) / pc_size[0]
        y_norm = (y - pc_min[1]) / pc_size[1]
        w_norm = w / pc_size[0]
        l_norm = l / pc_size[1]
        
        # 转换到特征图坐标 (注意：y对应行，x对应列)
        feat_x = x_norm * feature_size[1]  # 列
        feat_y = y_norm * feature_size[0]  # 行
        feat_w = w_norm * feature_size[1]
        feat_l = l_norm * feature_size[0]
        
        # 矩形框: [左上角x, 左上角y, 宽度, 高度]
        boxes_2d.append([
            feat_x - feat_w/2,  # 左上角x (列坐标)
            feat_y - feat_l/2,  # 左上角y (行坐标)
            feat_w,             # 宽度
            feat_l              # 高度
        ])
    
    return boxes_2d


def visualize_single_weight_map(weight_map, gt_boxes_2d=None, save_path='weight_viz.png', 
                               title='AQR Weight Map'):
    """可视化单个权重图"""
    plt.figure(figsize=(12, 10))
    
    # 自定义颜色映射
    colors = ['#000033', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('aqr_weights', colors, N=100)
    
    # 绘制权重图
    im = plt.imshow(weight_map, cmap=cmap, interpolation='bilinear', aspect='auto')
    plt.colorbar(im, label='Weight Value', fraction=0.046, pad=0.04)
    
    # 叠加GT框
    if gt_boxes_2d is not None and len(gt_boxes_2d) > 0:
        for i, box in enumerate(gt_boxes_2d):
            x, y, w, h = box
            rect = patches.Rectangle((x, y), w, h, linewidth=2.5, 
                                     edgecolor='lime', facecolor='none',
                                     label='GT Object' if i == 0 else '')
            plt.gca().add_patch(rect)
            
            # 在框中心添加编号
            cx, cy = x + w/2, y + h/2
            plt.text(cx, cy, str(i+1), color='white', fontsize=12, 
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='circle', facecolor='lime', alpha=0.7))
    
    # 添加统计信息
    info_text = f"Min: {weight_map.min():.4f} | Max: {weight_map.max():.4f} | Mean: {weight_map.mean():.4f}"
    plt.text(0.5, -0.05, info_text, transform=plt.gca().transAxes,
            ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if len(gt_boxes_2d) > 0:
        plt.legend(loc='upper right', fontsize=12)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Feature Width', fontsize=12)
    plt.ylabel('Feature Height', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def visualize_perspective_weights(weight_maps_pers, save_path='perspective_weights.png'):
    """可视化6个相机视角的权重图"""
    # weight_maps_pers shape: [bs, 6*40*100] 或 [6, 40, 100]
    if weight_maps_pers.dim() == 2:
        bs = weight_maps_pers.shape[0]
        weight_maps_pers = weight_maps_pers.view(bs, 6, 40, 100)[0]  # 取第一个batch
    elif weight_maps_pers.dim() == 3:
        weight_maps_pers = weight_maps_pers  # [6, 40, 100]
    
    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('AQR Perspective Weight Maps (All Camera Views)', 
                fontsize=16, fontweight='bold')
    
    colors = ['#000033', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('aqr_weights', colors, N=100)
    
    for view_idx in range(6):
        ax = axes[view_idx // 3, view_idx % 3]
        weight_map = weight_maps_pers[view_idx].cpu().numpy()
        
        im = ax.imshow(weight_map, cmap=cmap, interpolation='bilinear', aspect='auto')
        ax.set_title(f'{camera_names[view_idx]}\n[{weight_map.min():.3f}, {weight_map.max():.3f}]')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def analyze_weight_coverage(weight_map, gt_boxes_2d):
    """分析权重在GT目标上的覆盖情况"""
    H, W = weight_map.shape
    mask = np.zeros((H, W), dtype=bool)
    
    for box in gt_boxes_2d:
        x, y, w, h = [int(v) for v in box]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
        mask[y1:y2, x1:x2] = True
    
    total_weight = weight_map.sum()
    weight_in_boxes = weight_map[mask].sum()
    weight_outside = weight_map[~mask].sum()
    
    coverage_ratio = weight_in_boxes / total_weight if total_weight > 0 else 0
    
    print(f"\n📊 Weight Coverage Analysis:")
    print(f"   Total Weight: {total_weight:.4f}")
    print(f"   Weight in GT Boxes: {weight_in_boxes:.4f} ({coverage_ratio:.2%})")
    print(f"   Weight Outside: {weight_outside:.4f} ({1-coverage_ratio:.2%})")
    print(f"   Number of GT Objects: {len(gt_boxes_2d)}")
    
    return coverage_ratio


def main():
    parser = argparse.ArgumentParser(description='AQR权重快速可视化')
    parser.add_argument('--weight-file', required=True, help='保存的权重文件路径')
    parser.add_argument('--save-dir', default='aqr_visualization/', help='可视化结果保存目录')
    parser.add_argument('--sample-idx', type=int, default=0, help='批次中的样本索引')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载权重数据
    print(f"🔍 Loading weights from: {args.weight_file}")
    data = torch.load(args.weight_file)
    
    print(f"📦 Data keys: {list(data.keys())}")
    print(f"   Iteration: {data.get('iteration', 'N/A')}")
    
    # 提取数据
    weight_map_bev = data['weight_map_bev'][args.sample_idx].numpy()  # [180, 180]
    weight_map_pers = data['weight_map_pers'][args.sample_idx]  # [6*40*100] or [6, 40, 100]
    img_metas = data['img_metas'][args.sample_idx] if isinstance(data['img_metas'], list) else data['img_metas']
    
    print(f"   BEV weight map shape: {weight_map_bev.shape}")
    print(f"   Perspective weight map shape: {weight_map_pers.shape}")
    
    # 提取GT框
    gt_boxes_3d = img_metas.get('gt_bboxes_3d', None)
    if gt_boxes_3d is not None:
        gt_boxes_2d = project_3d_boxes_to_bev(gt_boxes_3d)
        print(f"   Found {len(gt_boxes_2d)} GT objects")
    else:
        gt_boxes_2d = []
        print(f"   ⚠️  No GT boxes found")
    
    # 可视化BEV权重图
    iter_num = data.get('iteration', 0)
    bev_save_path = os.path.join(args.save_dir, f'bev_weights_iter_{iter_num}_sample_{args.sample_idx}.png')
    visualize_single_weight_map(
        weight_map_bev, 
        gt_boxes_2d, 
        save_path=bev_save_path,
        title=f'AQR BEV Weight Map (Iter {iter_num}, Sample {args.sample_idx})'
    )
    
    # 分析权重覆盖
    if len(gt_boxes_2d) > 0:
        coverage_ratio = analyze_weight_coverage(weight_map_bev, gt_boxes_2d)
        
        if coverage_ratio < 0.3:
            print(f"   ⚠️  WARNING: Low weight coverage on GT objects ({coverage_ratio:.1%})")
            print(f"       AQR may not be focusing on targets properly!")
        elif coverage_ratio > 0.6:
            print(f"   ✅ GOOD: High weight coverage on GT objects ({coverage_ratio:.1%})")
            print(f"       AQR is successfully focusing on targets!")
    
    # 可视化透视权重图
    pers_save_path = os.path.join(args.save_dir, f'pers_weights_iter_{iter_num}_sample_{args.sample_idx}.png')
    visualize_perspective_weights(weight_map_pers, save_path=pers_save_path)
    
    print(f"\n✅ Visualization completed! Check {args.save_dir}")


if __name__ == '__main__':
    main()

