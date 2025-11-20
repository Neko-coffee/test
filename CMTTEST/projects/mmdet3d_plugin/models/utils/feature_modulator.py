# ------------------------------------------------------------------------
# FeatureModulator - 特征调制模块
# 核心功能：使用权重图对原始特征图进行逐元素调制
# 实现空间级别的模态重要性控制
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner.base_module import BaseModule
from mmdet.models.builder import NECKS


@NECKS.register_module()
class FeatureModulator(BaseModule):
    """
    特征调制模块
    
    使用生成的权重图对LiDAR和Camera特征图进行逐元素调制，
    实现基于权重的动态特征增强和抑制。
    
    Args:
        modulation_type (str): 调制类型 ['element_wise', 'channel_wise', 'adaptive']
        normalize_weights (bool): 是否对权重图进行归一化
        residual_connection (bool): 是否使用残差连接
        residual_weight (float): 残差连接权重
        learnable_modulation (bool): 是否使用可学习的调制参数
        activation (str): 激活函数类型 ['none', 'sigmoid', 'tanh', 'relu']
        init_cfg (dict): 初始化配置
    """
    
    def __init__(self,
                 modulation_type='element_wise',
                 normalize_weights=True,
                 residual_connection=True,
                 residual_weight=0.1,
                 learnable_modulation=False,
                 activation='none',
                 init_cfg=None):
        super(FeatureModulator, self).__init__(init_cfg=init_cfg)
        
        self.modulation_type = modulation_type
        self.normalize_weights = normalize_weights
        self.residual_connection = residual_connection
        self.residual_weight = residual_weight
        self.learnable_modulation = learnable_modulation
        self.activation = activation
        
        # 支持的调制类型
        self.supported_types = ['element_wise', 'channel_wise', 'adaptive']
        if modulation_type not in self.supported_types:
            raise ValueError(f"Unsupported modulation_type: {modulation_type}. "
                           f"Supported types: {self.supported_types}")
        
        # 可学习的调制参数
        if learnable_modulation:
            self.modulation_scale = nn.Parameter(torch.ones(1))
            self.modulation_bias = nn.Parameter(torch.zeros(1))
        
        # 激活函数
        self.activation_fn = self._get_activation_function(activation)
    
    def _get_activation_function(self, activation):
        """获取激活函数"""
        if activation == 'none':
            return nn.Identity()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'relu':
            return nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, features, weight_maps, feature_type='bev'):
        """
        前向传播：对特征图进行权重调制
        
        Args:
            features: [B, C, H, W] 或 [B, C, Views, H, W] 原始特征图
            weight_maps: [B, H, W] 或 [B, Views, H, W] 权重图
            feature_type: str 特征类型 ['bev', 'perspective']
            
        Returns:
            modulated_features: 调制后的特征图，形状与输入features相同
        """
        # 输入验证
        self._validate_inputs(features, weight_maps, feature_type)
        
        # 根据特征类型选择调制方法
        if feature_type == 'bev':
            return self._modulate_bev_features(features, weight_maps)
        elif feature_type == 'perspective':
            return self._modulate_perspective_features(features, weight_maps)
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")
    
    def _modulate_bev_features(self, features, weight_maps):
        """
        调制BEV特征图
        
        Args:
            features: [B, C, H, W] BEV特征图
            weight_maps: [B, H, W] BEV权重图
            
        Returns:
            modulated_features: [B, C, H, W] 调制后的BEV特征图
        """
        B, C, H, W = features.shape
        
        # 预处理权重图
        processed_weights = self._preprocess_weight_maps(weight_maps)
        
        # 根据调制类型进行调制
        if self.modulation_type == 'element_wise':
            modulated_features = self._element_wise_modulation(features, processed_weights)
        elif self.modulation_type == 'channel_wise':
            modulated_features = self._channel_wise_modulation(features, processed_weights)
        elif self.modulation_type == 'adaptive':
            modulated_features = self._adaptive_modulation(features, processed_weights)
        
        # 残差连接
        if self.residual_connection:
            modulated_features = modulated_features + self.residual_weight * features
        
        return modulated_features
    
    def _modulate_perspective_features(self, features, weight_maps):
        """
        调制透视特征图
        
        Args:
            features: [B*Views, C, H, W] 透视特征图
            weight_maps: [B, Views, H, W] 透视权重图
            
        Returns:
            modulated_features: [B*Views, C, H, W] 调制后的透视特征图
        """
        BV, C, H, W = features.shape
        B, Views, _, _ = weight_maps.shape
        
        # 重塑权重图以匹配特征图
        weight_maps_reshaped = weight_maps.view(B * Views, H, W)  # [B*Views, H, W]
        
        # 预处理权重图
        processed_weights = self._preprocess_weight_maps(weight_maps_reshaped)
        
        # 根据调制类型进行调制
        if self.modulation_type == 'element_wise':
            modulated_features = self._element_wise_modulation(features, processed_weights)
        elif self.modulation_type == 'channel_wise':
            modulated_features = self._channel_wise_modulation(features, processed_weights)
        elif self.modulation_type == 'adaptive':
            modulated_features = self._adaptive_modulation(features, processed_weights)
        
        # 残差连接
        if self.residual_connection:
            modulated_features = modulated_features + self.residual_weight * features
        
        return modulated_features
    
    def _preprocess_weight_maps(self, weight_maps):
        """预处理权重图"""
        processed_weights = weight_maps.clone()
        
        # 归一化
        if self.normalize_weights:
            # 按batch归一化到[0, 1]
            batch_size = processed_weights.shape[0]
            for b in range(batch_size):
                w_min = processed_weights[b].min()
                w_max = processed_weights[b].max()
                if w_max > w_min:
                    processed_weights[b] = (processed_weights[b] - w_min) / (w_max - w_min)
                else:
                    processed_weights[b] = torch.ones_like(processed_weights[b]) * 0.5
        
        # 可学习调制
        if self.learnable_modulation:
            processed_weights = processed_weights * self.modulation_scale + self.modulation_bias
        
        # 激活函数
        processed_weights = self.activation_fn(processed_weights)
        
        return processed_weights
    
    def _element_wise_modulation(self, features, weight_maps):
        """
        逐元素调制
        
        Args:
            features: [B, C, H, W] 特征图
            weight_maps: [B, H, W] 权重图
            
        Returns:
            modulated_features: [B, C, H, W] 调制后的特征图
        """
        # 广播乘法：[B, C, H, W] * [B, 1, H, W]
        weight_maps_expanded = weight_maps.unsqueeze(1)  # [B, 1, H, W]
        modulated_features = features * weight_maps_expanded
        
        return modulated_features
    
    def _channel_wise_modulation(self, features, weight_maps):
        """
        通道级调制
        
        Args:
            features: [B, C, H, W] 特征图
            weight_maps: [B, H, W] 权重图
            
        Returns:
            modulated_features: [B, C, H, W] 调制后的特征图
        """
        B, C, H, W = features.shape
        
        # 计算每个通道的权重
        # 这里使用简单的全局平均权重，可以根据需要扩展
        channel_weights = weight_maps.view(B, 1, H * W).mean(dim=2, keepdim=True)  # [B, 1, 1]
        channel_weights = channel_weights.view(B, 1, 1, 1).expand(B, C, 1, 1)
        
        # 应用通道级调制
        modulated_features = features * channel_weights
        
        return modulated_features
    
    def _adaptive_modulation(self, features, weight_maps):
        """
        自适应调制：结合逐元素和通道级调制
        
        Args:
            features: [B, C, H, W] 特征图
            weight_maps: [B, H, W] 权重图
            
        Returns:
            modulated_features: [B, C, H, W] 调制后的特征图
        """
        # 逐元素调制
        element_wise_result = self._element_wise_modulation(features, weight_maps)
        
        # 通道级调制
        channel_wise_result = self._channel_wise_modulation(features, weight_maps)
        
        # 自适应融合（这里使用简单的平均，可以扩展为可学习的融合）
        modulated_features = 0.7 * element_wise_result + 0.3 * channel_wise_result
        
        return modulated_features
    
    def _validate_inputs(self, features, weight_maps, feature_type):
        """输入验证"""
        if feature_type == 'bev':
            if features.dim() != 4:
                raise ValueError(f"BEV features should be 4D [B, C, H, W], got shape: {features.shape}")
            if weight_maps.dim() != 3:
                raise ValueError(f"BEV weight_maps should be 3D [B, H, W], got shape: {weight_maps.shape}")
            
            B, C, H, W = features.shape
            if weight_maps.shape != (B, H, W):
                raise ValueError(f"Weight map shape {weight_maps.shape} doesn't match feature shape {(B, H, W)}")
        
        elif feature_type == 'perspective':
            if features.dim() != 4:
                raise ValueError(f"Perspective features should be 4D [B*Views, C, H, W], got shape: {features.shape}")
            if weight_maps.dim() != 4:
                raise ValueError(f"Perspective weight_maps should be 4D [B, Views, H, W], got shape: {weight_maps.shape}")
            
            BV, C, H, W = features.shape
            B, Views, WH, WW = weight_maps.shape
            
            if BV != B * Views:
                raise ValueError(f"Feature batch size {BV} doesn't match weight map batch*views {B * Views}")
            if (H, W) != (WH, WW):
                raise ValueError(f"Feature spatial size {(H, W)} doesn't match weight map size {(WH, WW)}")
    
    def compute_modulation_statistics(self, weight_maps):
        """
        计算调制统计信息（用于调试和监控）
        
        Args:
            weight_maps: 权重图
            
        Returns:
            dict: 统计信息
        """
        stats = {}
        
        # 基本统计
        stats['mean'] = weight_maps.mean().item()
        stats['std'] = weight_maps.std().item()
        stats['min'] = weight_maps.min().item()
        stats['max'] = weight_maps.max().item()
        
        # 分布统计
        stats['zero_ratio'] = (weight_maps == 0).float().mean().item()
        stats['high_weight_ratio'] = (weight_maps > 0.8).float().mean().item()
        stats['low_weight_ratio'] = (weight_maps < 0.2).float().mean().item()
        
        # 空间统计
        if weight_maps.dim() >= 3:
            # 计算权重图的空间集中度
            spatial_variance = []
            for b in range(weight_maps.shape[0]):
                w = weight_maps[b] if weight_maps.dim() == 3 else weight_maps[b].flatten()
                spatial_variance.append(w.var().item())
            stats['spatial_variance_mean'] = np.mean(spatial_variance)
            stats['spatial_variance_std'] = np.std(spatial_variance)
        
        return stats
    
    def visualize_modulation_effect(self, original_features, modulated_features, weight_maps, 
                                  save_path="debug_modulation/", batch_idx=0, channel_idx=0):
        """
        可视化调制效果
        
        Args:
            original_features: 原始特征图
            modulated_features: 调制后特征图
            weight_maps: 权重图
            save_path: 保存路径
            batch_idx: 要可视化的batch索引
            channel_idx: 要可视化的通道索引
        """
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        # 提取要可视化的特征
        orig_feat = original_features[batch_idx, channel_idx].detach().cpu().numpy()
        mod_feat = modulated_features[batch_idx, channel_idx].detach().cpu().numpy()
        weight = weight_maps[batch_idx].detach().cpu().numpy()
        
        # 创建对比图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始特征
        im1 = axes[0, 0].imshow(orig_feat, cmap='viridis')
        axes[0, 0].set_title('Original Features')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 权重图
        im2 = axes[0, 1].imshow(weight, cmap='hot')
        axes[0, 1].set_title('Weight Map')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 调制后特征
        im3 = axes[0, 2].imshow(mod_feat, cmap='viridis')
        axes[0, 2].set_title('Modulated Features')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 差异图
        diff = mod_feat - orig_feat
        im4 = axes[1, 0].imshow(diff, cmap='RdBu_r')
        axes[1, 0].set_title('Difference (Mod - Orig)')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # 权重分布直方图
        axes[1, 1].hist(weight.flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_title('Weight Distribution')
        axes[1, 1].set_xlabel('Weight Value')
        axes[1, 1].set_ylabel('Frequency')
        
        # 调制强度分析
        modulation_ratio = np.abs(diff) / (np.abs(orig_feat) + 1e-8)
        im5 = axes[1, 2].imshow(modulation_ratio, cmap='plasma')
        axes[1, 2].set_title('Modulation Intensity')
        plt.colorbar(im5, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/modulation_effect_batch_{batch_idx}_ch_{channel_idx}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()


# 默认配置
DEFAULT_MODULATOR_CONFIG = dict(
    type='FeatureModulator',
    modulation_type='element_wise',
    normalize_weights=True,
    residual_connection=True,
    residual_weight=0.1,
    learnable_modulation=False,
    activation='none'
)
