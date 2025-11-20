# ------------------------------------------------------------------------
# WeightRenderer - 权重图渲染器
# 核心功能：将离散的per-query权重渲染到与特征图同尺寸的2D权重图上
# 实现空间级别的模态重要性控制
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from mmcv.runner.base_module import BaseModule
from mmdet.models.builder import NECKS
import warnings


@NECKS.register_module()
class WeightRenderer(BaseModule):
    """
    权重图渲染器
    
    将每个查询的权重值根据其在特征图中的位置"散布"到完整的权重图上，
    实现从查询级别到像素级别的权重传播。
    
    Args:
        render_method (str): 渲染方法 ['gaussian', 'bilinear', 'direct', 'distance_weighted']
        gaussian_sigma (float): 高斯核标准差
        bilinear_radius (float): 双线性插值半径
        distance_decay (float): 距离衰减因子
        min_weight_threshold (float): 最小权重阈值
        bev_feature_shape (tuple): BEV特征图形状 (H, W)
        pers_feature_shape (tuple): 透视特征图形状 (Views, H, W)
        normalize_weights (bool): 是否归一化权重图
        init_cfg (dict): 初始化配置
    """
    
    def __init__(self,
                 render_method='gaussian',
                 gaussian_sigma=2.0,
                 bilinear_radius=1.5,
                 distance_decay=0.8,
                 min_weight_threshold=0.01,
                 bev_feature_shape=(180, 180),
                 pers_feature_shape=(6, 40, 100),
                 normalize_weights=True,
                 max_weight_clamp=1.5,  # 🔥 新增：可配置的权重裁剪上限
                 init_cfg=None):
        super(WeightRenderer, self).__init__(init_cfg=init_cfg)
        
        self.render_method = render_method
        self.gaussian_sigma = gaussian_sigma
        self.bilinear_radius = bilinear_radius
        self.distance_decay = distance_decay
        self.min_weight_threshold = min_weight_threshold
        self.bev_feature_shape = bev_feature_shape
        self.pers_feature_shape = pers_feature_shape
        self.normalize_weights = normalize_weights
        self.max_weight_clamp = max_weight_clamp  # 🔥 保存裁剪上限
        
        # 支持的渲染方法
        self.supported_methods = ['gaussian', 'bilinear', 'direct', 'distance_weighted']
        if render_method not in self.supported_methods:
            raise ValueError(f"Unsupported render_method: {render_method}. "
                           f"Supported methods: {self.supported_methods}")
        
        # 预计算高斯核（如果使用高斯渲染）
        if render_method == 'gaussian':
            self._precompute_gaussian_kernel()
    
    def _precompute_gaussian_kernel(self):
        """预计算高斯核"""
        kernel_size = int(6 * self.gaussian_sigma + 1)  # 99.7%的高斯分布范围
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 创建2D高斯核
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        # 🔥 兼容PyTorch 1.9：不使用indexing参数（默认就是'ij'模式）
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.gaussian_sigma**2))
        kernel = kernel / kernel.sum()
        
        self.register_buffer('gaussian_kernel', kernel)
        self.kernel_size = kernel_size
    
    def render_bev_weights(self, query_weights, pts_bev, feature_shape=None):
        """
        将查询权重渲染到BEV特征图上
        
        Args:
            query_weights: [bs, num_queries] 查询权重
            pts_bev: [bs, num_queries, 2] BEV特征图坐标 (y, x)
            feature_shape: tuple BEV特征图形状，默认使用初始化参数
            
        Returns:
            weight_map: [bs, H, W] BEV权重图
        """
        if feature_shape is None:
            feature_shape = self.bev_feature_shape
        
        batch_size, num_queries = query_weights.shape
        H, W = feature_shape
        
        # 输入验证
        self._validate_inputs(query_weights, pts_bev, (H, W))
        
        # 初始化权重图
        weight_map = torch.zeros(batch_size, H, W, device=query_weights.device, dtype=query_weights.dtype)
        
        # 根据渲染方法选择实现
        if self.render_method == 'gaussian':
            weight_map = self._render_gaussian(weight_map, query_weights, pts_bev)
        elif self.render_method == 'bilinear':
            weight_map = self._render_bilinear(weight_map, query_weights, pts_bev)
        elif self.render_method == 'direct':
            weight_map = self._render_direct(weight_map, query_weights, pts_bev)
        elif self.render_method == 'distance_weighted':
            weight_map = self._render_distance_weighted(weight_map, query_weights, pts_bev)
        
        # 后处理
        weight_map = self._postprocess_weight_map(weight_map)
        
        return weight_map
    
    def render_perspective_weights(self, query_weights, pts_pers, feature_shape=None):
        """
        将查询权重渲染到透视特征图上
        
        Args:
            query_weights: [bs, num_queries] 查询权重
            pts_pers: [bs, num_queries, 3] 透视特征图坐标 (view, h, w)
            feature_shape: tuple 透视特征图形状，默认使用初始化参数
            
        Returns:
            weight_map: [bs, num_views, H, W] 透视权重图
        """
        if feature_shape is None:
            feature_shape = self.pers_feature_shape
        
        batch_size, num_queries = query_weights.shape
        num_views, H, W = feature_shape
        
        # 初始化权重图
        weight_map = torch.zeros(batch_size, num_views, H, W, 
                               device=query_weights.device, dtype=query_weights.dtype)
        
        # 分视角处理
        for view_idx in range(num_views):
            # 筛选当前视角的有效查询
            view_mask = (pts_pers[:, :, 0] == view_idx) & (~torch.isnan(pts_pers[:, :, 1]))
            
            if not view_mask.any():
                continue
            
            # 提取当前视角的坐标和权重
            view_coords = pts_pers[:, :, 1:3][view_mask]  # [valid_queries, 2] (h, w)
            view_weights = query_weights[view_mask]       # [valid_queries]
            
            if len(view_coords) == 0:
                continue
            
            # 获取对应的batch索引
            batch_indices = torch.arange(batch_size, device=query_weights.device)[:, None].expand(-1, num_queries)[view_mask]
            
            # 渲染到当前视角
            view_weight_map = self._render_to_single_view(
                view_weights, view_coords, batch_indices, batch_size, (H, W)
            )
            
            weight_map[:, view_idx] = view_weight_map
        
        # 🔥 修复：添加后处理（逐视角裁剪）
        # Bug: 之前缺少这一步，导致Camera权重图max=70.8而非1.5
        for view_idx in range(num_views):
            weight_map[:, view_idx] = self._postprocess_weight_map(weight_map[:, view_idx])
        
        return weight_map
    
    def _render_to_single_view(self, weights, coords, batch_indices, batch_size, feature_shape):
        """渲染到单个视角的特征图"""
        H, W = feature_shape
        weight_map = torch.zeros(batch_size, H, W, device=weights.device, dtype=weights.dtype)
        
        if self.render_method == 'gaussian':
            weight_map = self._render_gaussian_single_view(weight_map, weights, coords, batch_indices)
        elif self.render_method == 'direct':
            weight_map = self._render_direct_single_view(weight_map, weights, coords, batch_indices)
        # 其他方法可以类似实现
        
        return weight_map
    
    def _render_gaussian(self, weight_map, query_weights, pts_coords):
        """高斯核渲染"""
        batch_size, num_queries = query_weights.shape
        H, W = weight_map.shape[1], weight_map.shape[2]
        half_kernel = self.kernel_size // 2
        
        for b in range(batch_size):
            for q in range(num_queries):
                weight = query_weights[b, q].item()
                if weight < self.min_weight_threshold:
                    continue
                
                y, x = pts_coords[b, q]
                y, x = int(y.item()), int(x.item())
                
                # 边界检查
                if not (0 <= y < H and 0 <= x < W):
                    continue
                
                # 计算高斯核应用范围
                y_start = max(0, y - half_kernel)
                y_end = min(H, y + half_kernel + 1)
                x_start = max(0, x - half_kernel)
                x_end = min(W, x + half_kernel + 1)
                
                # 计算核的有效区域
                ky_start = half_kernel - (y - y_start)
                ky_end = ky_start + (y_end - y_start)
                kx_start = half_kernel - (x - x_start)
                kx_end = kx_start + (x_end - x_start)
                
                # 应用高斯核
                kernel_region = self.gaussian_kernel[ky_start:ky_end, kx_start:kx_end]
                weight_map[b, y_start:y_end, x_start:x_end] += weight * kernel_region
        
        return weight_map
    
    def _render_gaussian_single_view(self, weight_map, weights, coords, batch_indices):
        """单视角高斯核渲染"""
        H, W = weight_map.shape[1], weight_map.shape[2]
        half_kernel = self.kernel_size // 2
        
        for i in range(len(weights)):
            b = batch_indices[i].item()
            weight = weights[i].item()
            if weight < self.min_weight_threshold:
                continue
            
            y, x = coords[i]
            y, x = int(y.item()), int(x.item())
            
            # 边界检查
            if not (0 <= y < H and 0 <= x < W):
                continue
            
            # 应用高斯核（同上面的逻辑）
            y_start = max(0, y - half_kernel)
            y_end = min(H, y + half_kernel + 1)
            x_start = max(0, x - half_kernel)
            x_end = min(W, x + half_kernel + 1)
            
            ky_start = half_kernel - (y - y_start)
            ky_end = ky_start + (y_end - y_start)
            kx_start = half_kernel - (x - x_start)
            kx_end = kx_start + (x_end - x_start)
            
            kernel_region = self.gaussian_kernel[ky_start:ky_end, kx_start:kx_end]
            weight_map[b, y_start:y_end, x_start:x_end] += weight * kernel_region
        
        return weight_map
    
    def _render_bilinear(self, weight_map, query_weights, pts_coords):
        """双线性插值渲染"""
        batch_size, num_queries = query_weights.shape
        H, W = weight_map.shape[1], weight_map.shape[2]
        
        for b in range(batch_size):
            for q in range(num_queries):
                weight = query_weights[b, q].item()
                if weight < self.min_weight_threshold:
                    continue
                
                y, x = pts_coords[b, q]
                y, x = y.item(), x.item()
                
                # 双线性插值的四个邻近点
                y0, x0 = int(y), int(x)
                y1, x1 = y0 + 1, x0 + 1
                
                # 边界检查
                if not (0 <= y0 < H-1 and 0 <= x0 < W-1):
                    continue
                
                # 计算插值权重
                wy1, wx1 = y - y0, x - x0
                wy0, wx0 = 1 - wy1, 1 - wx1
                
                # 应用双线性插值
                weight_map[b, y0, x0] += weight * wy0 * wx0
                weight_map[b, y0, x1] += weight * wy0 * wx1
                weight_map[b, y1, x0] += weight * wy1 * wx0
                weight_map[b, y1, x1] += weight * wy1 * wx1
        
        return weight_map
    
    def _render_direct(self, weight_map, query_weights, pts_coords):
        """直接赋值渲染"""
        batch_size, num_queries = query_weights.shape
        H, W = weight_map.shape[1], weight_map.shape[2]
        
        for b in range(batch_size):
            for q in range(num_queries):
                weight = query_weights[b, q].item()
                if weight < self.min_weight_threshold:
                    continue
                
                y, x = pts_coords[b, q]
                y, x = int(y.item()), int(x.item())
                
                # 边界检查
                if 0 <= y < H and 0 <= x < W:
                    weight_map[b, y, x] = max(weight_map[b, y, x], weight)  # 取最大值避免覆盖
        
        return weight_map
    
    def _render_direct_single_view(self, weight_map, weights, coords, batch_indices):
        """单视角直接赋值渲染"""
        H, W = weight_map.shape[1], weight_map.shape[2]
        
        for i in range(len(weights)):
            b = batch_indices[i].item()
            weight = weights[i].item()
            if weight < self.min_weight_threshold:
                continue
            
            y, x = coords[i]
            y, x = int(y.item()), int(x.item())
            
            # 边界检查
            if 0 <= y < H and 0 <= x < W:
                weight_map[b, y, x] = max(weight_map[b, y, x], weight)
        
        return weight_map
    
    def _render_distance_weighted(self, weight_map, query_weights, pts_coords):
        """距离加权渲染"""
        batch_size, num_queries = query_weights.shape
        H, W = weight_map.shape[1], weight_map.shape[2]
        max_distance = min(H, W) * 0.2  # 最大影响距离
        
        # 创建坐标网格
        # 🔥 兼容PyTorch 1.9：不使用indexing参数
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=weight_map.device),
            torch.arange(W, device=weight_map.device)
        )
        
        for b in range(batch_size):
            for q in range(num_queries):
                weight = query_weights[b, q].item()
                if weight < self.min_weight_threshold:
                    continue
                
                y, x = pts_coords[b, q]
                y, x = y.item(), x.item()
                
                # 计算距离
                distances = torch.sqrt((y_grid - y)**2 + (x_grid - x)**2)
                
                # 距离衰减权重
                distance_weights = torch.exp(-distances / (max_distance * self.distance_decay))
                distance_weights[distances > max_distance] = 0
                
                # 应用权重
                weight_map[b] += weight * distance_weights
        
        return weight_map
    
    def _validate_inputs(self, query_weights, pts_coords, feature_shape):
        """输入验证"""
        batch_size, num_queries = query_weights.shape
        
        if pts_coords.shape != (batch_size, num_queries, 2):
            raise ValueError(f"pts_coords shape mismatch. Expected: {(batch_size, num_queries, 2)}, "
                           f"Got: {pts_coords.shape}")
        
        if torch.any(query_weights < 0) or torch.any(query_weights > 1):
            warnings.warn("Query weights should be in range [0, 1]. Values will be clamped.")
            query_weights.clamp_(0, 1)
    
    def _postprocess_weight_map(self, weight_map):
        """
        权重图后处理
        
        🔥 关键修复：
        1. 不进行全局归一化（避免破坏原始权重幅值）
        2. 只应用阈值过滤（去除噪声）
        3. 可选：轻度裁剪防止极端值（而非归一化）
        """
        # 应用最小阈值
        weight_map[weight_map < self.min_weight_threshold] = 0
        
        # 🔥 修复：使用裁剪而非归一化
        # 原因：归一化会破坏query权重的绝对数值信息
        # 目标：保留原始权重幅值，只防止极端爆炸
        if self.normalize_weights:
            # 🔥 使用可配置的裁剪上限（默认1.5，可通过max_weight_clamp调整）
            # 假设原始query权重在[0,1]，高斯叠加后合理上限是1.5-3.0
            weight_map = torch.clamp(weight_map, min=0, max=self.max_weight_clamp)
        
        return weight_map
    
    def visualize_weight_maps(self, weight_maps, save_path="debug_weights/", prefix="weight_map"):
        """
        可视化权重图
        
        Args:
            weight_maps: [bs, H, W] 或 [bs, num_views, H, W] 权重图
            save_path: 保存路径
            prefix: 文件名前缀
        """
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        if weight_maps.dim() == 3:  # BEV weight maps
            for b in range(weight_maps.shape[0]):
                plt.figure(figsize=(8, 8))
                plt.imshow(weight_maps[b].cpu().numpy(), cmap='hot', interpolation='bilinear')
                plt.title(f'Batch {b} - {prefix}')
                plt.colorbar()
                plt.savefig(f'{save_path}/{prefix}_batch_{b}.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        elif weight_maps.dim() == 4:  # Perspective weight maps
            for b in range(weight_maps.shape[0]):
                num_views = weight_maps.shape[1]
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for v in range(min(num_views, 6)):
                    im = axes[v].imshow(weight_maps[b, v].cpu().numpy(), cmap='hot', interpolation='bilinear')
                    axes[v].set_title(f'View {v}')
                    plt.colorbar(im, ax=axes[v])
                
                plt.suptitle(f'Batch {b} - {prefix} (All Views)')
                plt.tight_layout()
                plt.savefig(f'{save_path}/{prefix}_batch_{b}_all_views.png', dpi=150, bbox_inches='tight')
                plt.close()


# 默认配置
DEFAULT_RENDERER_CONFIG = dict(
    type='WeightRenderer',
    render_method='gaussian',
    gaussian_sigma=2.0,
    bilinear_radius=1.5,
    distance_decay=0.8,
    min_weight_threshold=0.01,
    bev_feature_shape=(180, 180),
    pers_feature_shape=(6, 40, 100),
    normalize_weights=True,
    max_weight_clamp=1.5  # 🔥 默认裁剪上限
)
