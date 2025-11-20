# Copyright (c) 2024 CMT-AQR Team. All rights reserved.
"""
Attention Bias Generator for Local Spatial Modulation
局部空间注意力bias生成器

功能：
1. 根据query的空间投影位置生成局部窗口
2. 将per-query权重扩散到局部窗口内的特征
3. 生成细粒度的attention bias矩阵用于Transformer

设计思路：
- 不直接修改特征图，而是影响attention计算
- 保持特征语义不变，只调整query对不同区域的关注程度
- 使用局部窗口而非全局，提供空间先验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32
import numpy as np


class AttentionBiasGenerator(BaseModule):
    """
    局部注意力Bias生成器
    
    核心功能：
    - 输入：per-query权重 + 空间位置
    - 输出：[bs, num_queries, num_features] 的bias矩阵
    
    实现策略：
    - 向量化计算，避免循环
    - 局部窗口控制bias范围
    - 支持BEV和Camera两种特征图
    """
    
    def __init__(self,
                 bev_feature_shape=(180, 180),      # 🔥 1600x640默认: (180, 180)
                 pers_feature_shape=(6, 40, 100),   # 🔥 1600x640默认: (6, 40, 100)
                 window_size=15,                    # 🔥 1600x640默认: 15
                 bias_scale=2.5,
                 learnable_scale=True,              # 🔥 默认启用可学习scale
                 min_scale=0.5,
                 max_scale=5.0,
                 use_local_bias=True,
                 use_gaussian_window=False,
                 gaussian_sigma=2.5,                # 🔥 1600x640默认: 2.5
                 debug_print=True,                  # 🔥 默认启用调试打印
                 print_interval=1000,               # 🔥 默认1000个iteration打印一次
                 fp16=True,                         # 🔥 默认启用FP16
                 init_cfg=None):
        """
        Args:
            bev_feature_shape (tuple): BEV特征图尺寸 (H, W)
            pers_feature_shape (tuple): 透视特征图尺寸 (num_views, H, W)
            window_size (int): 局部窗口大小（正方形窗口的边长）
            bias_scale (float): bias缩放因子的初始值
            learnable_scale (bool): 是否让bias_scale可学习
            min_scale (float): bias_scale的最小值（防止退化）
            max_scale (float): bias_scale的最大值（防止softmax饱和）
            use_local_bias (bool): True=局部窗口bias, False=全局bias
            use_gaussian_window (bool): True=高斯衰减窗口, False=均匀窗口（仅当use_local_bias=True时生效）
            gaussian_sigma (float): 高斯核的标准差（仅当use_gaussian_window=True时生效）
            debug_print (bool): 是否打印调试信息（bias统计、权重分布等）
            print_interval (int): 打印间隔（每N个iteration打印一次），仅当debug_print=True时生效
            fp16 (bool): 是否使用半精度存储bias矩阵
            init_cfg (dict): 初始化配置
        """
        super(AttentionBiasGenerator, self).__init__(init_cfg=init_cfg)
        
        self.bev_h, self.bev_w = bev_feature_shape
        self.num_views, self.pers_h, self.pers_w = pers_feature_shape
        self.window_size = window_size
        self.use_local_bias = use_local_bias
        self.use_gaussian_window = use_gaussian_window
        self.gaussian_sigma = gaussian_sigma
        self.debug_print = debug_print
        self.print_interval = print_interval
        self._iter_count = 0  # 迭代计数器
        self.fp16 = fp16
        self.learnable_scale = learnable_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        # 🔥 可学习的bias_scale
        if learnable_scale:
            # 🔥 修复：使用 FloatTensor 并明确指定 dtype，避免梯度计算问题
            self.bias_scale = nn.Parameter(torch.tensor([bias_scale], dtype=torch.float32))
        else:
            self.register_buffer('bias_scale', torch.tensor([bias_scale], dtype=torch.float32))
        
        # 预计算窗口偏移量（加速）
        self._init_window_offsets()
        
        print(f"✅ AttentionBiasGenerator initialized:")
        print(f"   BEV shape: {bev_feature_shape}")
        print(f"   Pers shape: {pers_feature_shape}")
        print(f"   Window size: {window_size} ({'local' if use_local_bias else 'global'})")
        print(f"   Bias scale: {bias_scale} ({'learnable' if learnable_scale else 'fixed'})")
        if learnable_scale:
            print(f"   Scale range: [{min_scale}, {max_scale}]")
        print(f"   FP16: {fp16}")
    
    def _init_window_offsets(self):
        """预计算窗口偏移量"""
        # 🔥 确保window_size就是实际窗口大小（例如window_size=8 → 8x8窗口）
        # 窗口范围：[-half_window+1, half_window]，共window_size个元素
        half_window = self.window_size // 2
        if self.window_size % 2 == 0:
            # 偶数窗口：例如8 → [-3, -2, -1, 0, 1, 2, 3, 4]
            offsets = torch.arange(-half_window + 1, half_window + 1)
        else:
            # 奇数窗口：例如9 → [-4, -3, -2, -1, 0, 1, 2, 3, 4]
            offsets = torch.arange(-half_window, half_window + 1)
        
        # 2D网格偏移（用于BEV）
        # 兼容旧版PyTorch（<1.10）：不使用indexing参数
        try:
            y_offsets, x_offsets = torch.meshgrid(offsets, offsets, indexing='ij')
        except TypeError:
            # PyTorch < 1.10：默认就是'ij'索引方式
            y_offsets, x_offsets = torch.meshgrid(offsets, offsets)
        self.register_buffer('y_offsets', y_offsets.reshape(-1))  # [window_size^2]
        self.register_buffer('x_offsets', x_offsets.reshape(-1))  # [window_size^2]
        
        # 1D索引偏移（用于展平后的特征）
        window_offsets_bev = y_offsets * self.bev_w + x_offsets
        self.register_buffer('window_offsets_bev', window_offsets_bev.reshape(-1))  # [window_size^2]
        
        window_offsets_pers = y_offsets * self.pers_w + x_offsets
        self.register_buffer('window_offsets_pers', window_offsets_pers.reshape(-1))  # [window_size^2]
        
        # 🔥 预计算高斯权重（如果启用）
        if self.use_gaussian_window:
            # 计算窗口内每个位置到中心的距离
            distances = torch.sqrt(self.y_offsets.float()**2 + self.x_offsets.float()**2)
            # [window_size^2]
            
            # 高斯衰减：exp(-distance^2 / (2 * sigma^2))
            gaussian_weights = torch.exp(-distances**2 / (2 * self.gaussian_sigma**2))
            # 归一化（可选，让总和为1）
            # gaussian_weights = gaussian_weights / gaussian_weights.sum()
            
            self.register_buffer('gaussian_weights', gaussian_weights)  # [window_size^2]
        else:
            # 均匀权重
            uniform_weights = torch.ones(self.window_size**2)
            self.register_buffer('gaussian_weights', uniform_weights)
    
    @force_fp32(apply_to=('lidar_weights', 'camera_weights'))
    def forward(self,
                lidar_weights,      # [bs, num_queries] LiDAR模态权重
                camera_weights,     # [bs, num_queries] Camera模态权重
                pts_bev_indices,    # [bs, num_queries] BEV特征图位置索引
                pts_pers_indices):  # [bs, num_queries, 3] 透视特征图位置索引 (view, h, w)
        """
        生成局部attention bias
        
        Args:
            lidar_weights: [bs, num_queries] AQR生成的LiDAR权重
            camera_weights: [bs, num_queries] AQR生成的Camera权重
            pts_bev_indices: [bs, num_queries] query在BEV特征图中的位置（1D索引）
            pts_pers_indices: [bs, num_queries, 3] query在透视特征图中的位置（view, h, w）
        
        Returns:
            attention_bias: [bs, num_queries, total_features]
                其中 total_features = bev_h*bev_w + num_views*pers_h*pers_w
                
        示例：
            bs=2, num_queries=900
            bev_features = 180*180 = 32400
            pers_features = 6*40*100 = 24000
            total_features = 56400
            
            输出：[2, 900, 56400]
        """
        batch_size, num_queries = lidar_weights.shape
        device = lidar_weights.device
        
        # 🔥 权重已经是[-1, 1]范围（来自AQRWeightGenerator的tanh输出）
        # weight=+1.0 → 正bias → 增强attention → 红色
        # weight=-1.0 → 负bias → 抑制attention → 蓝色
        
        # 1. 生成BEV bias（直接使用权重，已经是[-1, 1]范围）
        bev_bias = self._generate_bev_bias(
            lidar_weights,      # [bs, num_queries], 范围[-1, 1]
            pts_bev_indices     # [bs, num_queries]
        )  # → [bs, num_queries, bev_h*bev_w], 范围[-1, 1]
        
        # 2. 生成Camera bias（直接使用权重，已经是[-1, 1]范围）
        camera_bias = self._generate_camera_bias(
            camera_weights,     # [bs, num_queries], 范围[-1, 1]
            pts_pers_indices    # [bs, num_queries, 3]
        )  # → [bs, num_queries, num_views*pers_h*pers_w], 范围[-1, 1]
        
        # 3. 拼接成完整的bias矩阵
        attention_bias = torch.cat([bev_bias, camera_bias], dim=-1)
        # → [bs, num_queries, total_features], 范围[-1, 1]
        
        # 4. 🔥 应用缩放因子（带约束）
        if self.learnable_scale:
            # 🔥 Step 1: Clamp scale到安全范围
            scale = torch.clamp(self.bias_scale[0], min=self.min_scale, max=self.max_scale)  # 取第一个元素
            
        else:
            scale = self.bias_scale[0]  # 取第一个元素
        
        attention_bias = attention_bias * scale
        # 🔥 权重已经是[-1, 1]，乘以scale后bias是[-scale, +scale]
        # weight=+1.0 → bias=+scale（正bias，增强attention，红色）
        # weight=-1.0 → bias=-scale（负bias，抑制attention，蓝色）
        # 例如：scale=3.0 → bias范围[-3.0, +3.0]
        
        # 5. 🔥 实时监控bias统计信息（可配置）
        if self.debug_print:  # 🔥 移除training限制，训练和验证都打印
            self._iter_count += 1
            if self._iter_count % self.print_interval == 0:
                # 🔥 在 no_grad() 之前保存梯度信息（因为梯度可能在后续被清零）
                grad_value = None
                if self.learnable_scale and self.bias_scale.grad is not None:
                    grad_value = self.bias_scale.grad[0].item()
                
                with torch.no_grad():
                    # 分离BEV和Camera部分
                    bev_part = attention_bias[:, :, :self.bev_h * self.bev_w]
                    cam_part = attention_bias[:, :, self.bev_h * self.bev_w:]
                    
                    print(f"\n{'='*70}")
                    print(f"📊 [AttentionBias] Monitoring Report (Iter {self._iter_count}):")
                    
                    # Scale信息
                    if self.learnable_scale:
                        current_scale = scale.item()
                        print(f"   🔧 Learnable Bias Scale: {current_scale:.4f} (range: [{self.min_scale}, {self.max_scale}])")
                        
                        # 🔥 打印保存的梯度信息
                        if grad_value is not None:
                            print(f"      📉 Gradient: {grad_value:+.8f}")
                        else:
                            print(f"      ⚠️  Gradient: None (未计算或已清零)")
                        
                        # 检查 requires_grad
                        print(f"      🔍 requires_grad: {self.bias_scale.requires_grad}")
                        
                        if current_scale > 0.9 * self.max_scale:
                            print(f"      ⚠️  WARNING: Scale接近上限 ({current_scale:.4f} / {self.max_scale})!")
                        elif current_scale < 1.1 * self.min_scale:
                            print(f"      ⚠️  WARNING: Scale接近下限 ({current_scale:.4f} / {self.min_scale})!")
                    else:
                        print(f"   🔧 Fixed Bias Scale: {self.bias_scale:.4f}")
                    
                    # Bias统计信息
                    print(f"   📈 Bias Statistics:")
                    print(f"      Overall  - Mean: {attention_bias.mean().item():+.4f}, Std: {attention_bias.std().item():.4f}")
                    print(f"                 Range: [{attention_bias.min().item():+.4f}, {attention_bias.max().item():+.4f}]")
                    print(f"      BEV      - Mean: {bev_part.mean().item():+.4f}, Std: {bev_part.std().item():.4f}")
                    print(f"      Camera   - Mean: {cam_part.mean().item():+.4f}, Std: {cam_part.std().item():.4f}")
                    
                    # 分布分析
                    positive_ratio = (attention_bias > 0).float().mean().item()
                    strong_positive = (attention_bias > 2.0).float().mean().item()
                    strong_negative = (attention_bias < -2.0).float().mean().item()
                    near_zero = (attention_bias.abs() < 0.1).float().mean().item()
                    print(f"      Distribution:")
                    print(f"         Positive: {positive_ratio*100:.1f}% | Negative: {(1-positive_ratio)*100:.1f}%")
                    print(f"         Strong+ (>+2): {strong_positive*100:.2f}% | Strong- (<-2): {strong_negative*100:.2f}%")
                    print(f"         Near-zero (|bias|<0.1): {near_zero*100:.1f}%")
                    
                    # 输入权重分析
                    print(f"   📊 Input Weights (from AQR):")
                    print(f"      LiDAR    - Mean: {lidar_weights.mean().item():+.4f}, Std: {lidar_weights.std().item():.4f}")
                    print(f"                 Range: [{lidar_weights.min().item():+.4f}, {lidar_weights.max().item():+.4f}]")
                    print(f"      Camera   - Mean: {camera_weights.mean().item():+.4f}, Std: {camera_weights.std().item():.4f}")
                    print(f"                 Range: [{camera_weights.min().item():+.4f}, {camera_weights.max().item():+.4f}]")
                    
                # 模态偏好分析
                lidar_prefer_ratio = (lidar_weights > camera_weights).float().mean().item()
                camera_prefer_ratio = (camera_weights > lidar_weights).float().mean().item()
                balanced_ratio = ((lidar_weights - camera_weights).abs() < 0.2).float().mean().item()
                
                print(f"      Modality Preference (per query):")
                print(f"         Prefer LiDAR: {lidar_prefer_ratio*100:.1f}% (lidar_w > camera_w)")
                print(f"         Prefer Camera: {camera_prefer_ratio*100:.1f}% (camera_w > lidar_w)")
                print(f"         Balanced: {balanced_ratio*100:.1f}% (|diff| < 0.2)")
                print(f"{'='*70}\n")
        
        # 5. 🔥 双重保险：裁剪最终bias范围
        # 即使scale被约束，仍然clamp一次确保不会超出softmax敏感区间
        # Softmax敏感区间：[-3, +3]最优，[-5, +5]安全
        max_bias = min(5.0, self.max_scale)  # 取max_scale和5.0的较小值
        attention_bias = torch.clamp(attention_bias, min=-max_bias, max=max_bias)
        
        # 6. 转换为fp16（如果需要）
        if self.fp16:
            attention_bias = attention_bias.half()
        
        return attention_bias
    
    def _generate_bev_bias(self, weights, positions):
        """
        生成BEV特征图的局部bias
        
        Args:
            weights: [bs, num_queries] 权重
            positions: [bs, num_queries] 1D位置索引
        
        Returns:
            bias: [bs, num_queries, bev_h*bev_w]
        """
        batch_size, num_queries = weights.shape
        device = weights.device
        total_features = self.bev_h * self.bev_w
        
        if not self.use_local_bias:
            # 全局bias：每个query对所有BEV特征施加相同bias
            return weights.unsqueeze(-1).expand(batch_size, num_queries, total_features)
        
        # === 局部bias：向量化实现 ===
        
        # 1. 计算所有query的局部窗口索引
        # positions: [bs, num_queries]
        # window_offsets_bev: [window_size^2]
        query_indices = positions.unsqueeze(-1) + self.window_offsets_bev.unsqueeze(0).unsqueeze(0)
        # → [bs, num_queries, window_size^2]
        
        # 2. 边界检查（2D网格）
        query_y = torch.div(positions, self.bev_w, rounding_mode='floor').long()  # [bs, num_queries]
        query_x = positions % self.bev_w   # [bs, num_queries]
        
        window_y = query_y.unsqueeze(-1) + self.y_offsets.unsqueeze(0).unsqueeze(0)
        window_x = query_x.unsqueeze(-1) + self.x_offsets.unsqueeze(0).unsqueeze(0)
        # → [bs, num_queries, window_size^2]
        
        valid_y = (window_y >= 0) & (window_y < self.bev_h)
        valid_x = (window_x >= 0) & (window_x < self.bev_w)
        valid_mask = valid_y & valid_x  # [bs, num_queries, window_size^2]
        
        # 3. 创建bias矩阵
        bias = torch.zeros(batch_size, num_queries, total_features, 
                          device=device, dtype=weights.dtype)
        
        # 4. 扩展权重到窗口
        weights_expanded = weights.unsqueeze(-1).expand(-1, -1, self.window_size**2)
        # → [bs, num_queries, window_size^2]
        
        # 🔥 4.5. 应用高斯衰减（如果启用）
        if self.use_gaussian_window:
            # gaussian_weights: [window_size^2]
            # weights_expanded: [bs, num_queries, window_size^2]
            weights_expanded = weights_expanded * self.gaussian_weights.unsqueeze(0).unsqueeze(0)
        
        # 5. 应用valid mask
        weights_masked = torch.where(valid_mask, weights_expanded, 
                                     torch.zeros_like(weights_expanded))
        
        # 6. 向量化填充（使用scatter_add）
        # 将窗口内的索引clip到有效范围
        query_indices_clamped = query_indices.clamp(0, total_features - 1).long()  # 🔥 强制转换为int64
        
        # 逐batch处理（避免scatter_add的维度问题）
        for b in range(batch_size):
            bias[b].scatter_add_(
                dim=1,  # 在feature维度scatter
                index=query_indices_clamped[b],  # [num_queries, window_size^2]
                src=weights_masked[b]             # [num_queries, window_size^2]
            )
        
        return bias
    
    def _generate_camera_bias(self, weights, positions):
        """
        生成Camera透视特征图的局部bias
        
        Args:
            weights: [bs, num_queries] 权重
            positions: [bs, num_queries, 3] 3D位置索引 (view, h, w)
        
        Returns:
            bias: [bs, num_queries, num_views*pers_h*pers_w]
        """
        batch_size, num_queries = weights.shape
        device = weights.device
        total_features = self.num_views * self.pers_h * self.pers_w
        
        if not self.use_local_bias:
            # 全局bias
            return weights.unsqueeze(-1).expand(batch_size, num_queries, total_features)
        
        # === 局部bias：向量化实现 ===
        
        # 1. 解析3D位置
        view_indices = positions[..., 0].long()  # [bs, num_queries]
        h_indices = positions[..., 1].long()     # [bs, num_queries]
        w_indices = positions[..., 2].long()     # [bs, num_queries]
        
        # 2. 计算1D索引
        positions_1d = (view_indices * self.pers_h * self.pers_w + 
                       h_indices * self.pers_w + 
                       w_indices)  # [bs, num_queries]
        
        # 3. 计算窗口索引（只在同一视角内）
        query_indices = positions_1d.unsqueeze(-1) + self.window_offsets_pers.unsqueeze(0).unsqueeze(0)
        # → [bs, num_queries, window_size^2]
        
        # 4. 边界检查（2D网格 + 视角一致性）
        window_h = h_indices.unsqueeze(-1) + self.y_offsets.unsqueeze(0).unsqueeze(0)
        window_w = w_indices.unsqueeze(-1) + self.x_offsets.unsqueeze(0).unsqueeze(0)
        # → [bs, num_queries, window_size^2]
        
        valid_h = (window_h >= 0) & (window_h < self.pers_h)
        valid_w = (window_w >= 0) & (window_w < self.pers_w)
        
        # 确保窗口不跨视角（检查窗口索引是否在同一视角内）
        window_view = torch.div(query_indices, self.pers_h * self.pers_w, rounding_mode='floor').long()
        valid_view = (window_view == view_indices.unsqueeze(-1))
        
        valid_mask = valid_h & valid_w & valid_view  # [bs, num_queries, window_size^2]
        
        # 5. 创建bias矩阵
        bias = torch.zeros(batch_size, num_queries, total_features,
                          device=device, dtype=weights.dtype)
        
        # 6. 扩展权重到窗口
        weights_expanded = weights.unsqueeze(-1).expand(-1, -1, self.window_size**2)
        
        # 🔥 6.5. 应用高斯衰减（如果启用）
        if self.use_gaussian_window:
            weights_expanded = weights_expanded * self.gaussian_weights.unsqueeze(0).unsqueeze(0)
        
        weights_masked = torch.where(valid_mask, weights_expanded,
                                     torch.zeros_like(weights_expanded))
        
        # 7. 向量化填充
        query_indices_clamped = query_indices.clamp(0, total_features - 1).long()  # 🔥 强制转换为int64
        
        for b in range(batch_size):
            bias[b].scatter_add_(
                dim=1,
                index=query_indices_clamped[b],
                src=weights_masked[b]
            )
        
        return bias
    
    def get_memory_usage(self, batch_size, num_queries):
        """
        估算内存占用
        
        Args:
            batch_size: batch大小
            num_queries: query数量
        
        Returns:
            dict: 内存使用信息
        """
        total_features = self.bev_h * self.bev_w + self.num_views * self.pers_h * self.pers_w
        
        # bias矩阵大小
        bias_elements = batch_size * num_queries * total_features
        bias_memory_fp32 = bias_elements * 4 / (1024**2)  # MB
        bias_memory_fp16 = bias_elements * 2 / (1024**2)  # MB
        
        return {
            'total_features': total_features,
            'bias_shape': (batch_size, num_queries, total_features),
            'memory_fp32_mb': bias_memory_fp32,
            'memory_fp16_mb': bias_memory_fp16,
            'current_dtype': 'fp16' if self.fp16 else 'fp32',
            'estimated_memory_mb': bias_memory_fp16 if self.fp16 else bias_memory_fp32
        }


# === 默认配置 ===
DEFAULT_ATTENTION_BIAS_CONFIG = dict(
    type='AttentionBiasGenerator',
    bev_feature_shape=(180, 180),
    pers_feature_shape=(6, 40, 100),
    window_size=15,                # 推荐值：与LAM的camera window一致
    bias_scale=1.0,                # 初始建议：1.0（无缩放）
    use_local_bias=True,           # 推荐：True（局部bias）
    fp16=True                      # 推荐：True（节省内存）
)

