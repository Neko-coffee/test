# ------------------------------------------------------------------------
# AQR Weight Generator - 基于MoME的AQR改造为权重图渲染机制
# 核心修改：从离散模态选择改为连续权重生成
# ------------------------------------------------------------------------
import copy
import numpy as np
import torch
import torch.nn as nn
import warnings
from einops import rearrange
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER


@TRANSFORMER.register_module()
class AQRWeightGenerator(BaseModule):
    """
    AQR权重生成器 - 基于MoME的AQR机制改造
    
    核心功能：
    1. 保留MoME的3D投影和局部注意力掩码（LAM）机制
    2. 将离散的模态选择改为连续的权重生成
    3. 为每个Query生成LiDAR和Camera的权重值
    
    Args:
        embed_dims (int): 嵌入维度，默认256
        encoder_config (dict): 编码器配置
        window_sizes (list): 局部注意力窗口大小 [camera_window, lidar_window]
        use_type_embed (bool): 是否使用模态类型嵌入
        pc_range (list): 点云范围
    """
    
    def __init__(self,
                 embed_dims=256,
                 encoder_config=None,
                 window_sizes=[15, 5],
                 use_type_embed=True,
                 pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
                 bev_feature_shape=(180, 180),  # 🔥 BEV特征图尺寸（可配置）
                 pers_feature_shape=(6, 40, 100),  # 🔥 透视特征图尺寸（可配置）
                 init_cfg=None):
        super(AQRWeightGenerator, self).__init__(init_cfg=init_cfg)
        
        self.embed_dims = embed_dims
        self.window_sizes = window_sizes
        self.use_type_embed = use_type_embed
        self.pc_range = pc_range
        self.bev_feature_shape = bev_feature_shape  # 🔥 保存BEV尺寸
        self.pers_feature_shape = pers_feature_shape  # 🔥 保存透视尺寸
        
        # 🔥 从配置中提取尺寸参数
        self.bev_h, self.bev_w = bev_feature_shape
        self.num_views, self.pers_h, self.pers_w = pers_feature_shape
        
        # 构建编码器（保留MoME的结构）
        if encoder_config is not None:
            self.encoder = build_transformer_layer_sequence(encoder_config)
            self.e_num_heads = self.encoder.layers[0].attentions[0].num_heads
        else:
            raise ValueError("encoder_config is required for AQRWeightGenerator")
            
        # 🔥 核心修改：从3类分类改为2个连续权重输出
        self.weight_predictor = nn.Linear(embed_dims, 2)  # [lidar_weight, camera_weight]
        
        # 🔥 关键修改：初始化偏置使初始权重接近0.82（温和增强）
        # sigmoid(1.5) ≈ 0.82 → 配合residual=0.7，初期增强40%，保护预训练特征
        nn.init.constant_(self.weight_predictor.bias, 1.5)  
        
        # 类型嵌入（保留MoME设计）
        if self.use_type_embed:
            self.bev_type_embed = nn.Parameter(torch.randn(embed_dims))
            self.rv_type_embed = nn.Parameter(torch.randn(embed_dims))
    
    def init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        
        # 🔥 再次确保weight_predictor的偏置初始化正确
        if hasattr(self, 'weight_predictor'):
            nn.init.constant_(self.weight_predictor.bias, 1.5)
        
        self._is_init = True
    
    def project_3d_to_features(self, ref_points, img_metas):
        """
        3D参考点投影到特征图坐标
        保留MoME的精确投影逻辑
        
        Args:
            ref_points: [bs, num_queries, 3] 归一化的3D参考点
            img_metas: 图像元数据
            
        Returns:
            pts_bev: [bs, num_queries, 2] BEV特征图坐标 (y, x)
            pts_pers: [bs, num_queries, 3] 透视特征图坐标 (view, h, w)
            pts_idx: [bs, num_queries] BEV特征图索引
            pts_pers_idx: [bs, num_queries] 透视特征图索引
        """
        # 反归一化到真实3D坐标
        _ref_points = ref_points.new_zeros(ref_points.shape)
        _ref_points[..., 0:1] = ref_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        _ref_points[..., 1:2] = ref_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        _ref_points[..., 2:3] = ref_points[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        
        # 获取相机投影矩阵
        _matrices = np.stack([np.stack(i['lidar2img']) for i in img_metas])
        _matrices = torch.tensor(_matrices).float().to(ref_points.device)
        batch, _num_queries, _ = _ref_points.shape
        
        # 3D → 2D 透视投影
        _position_4d = torch.cat([_ref_points, torch.ones((batch, _num_queries, 1), device=ref_points.device)], dim=-1)
        pts_2d = torch.einsum('bni,bvij->bvnj', _position_4d, _matrices.transpose(2,3))
        pts_2d[..., 2] = torch.clip(pts_2d[..., 2], min=1e-5, max=99999)
        pts_2d[..., 0] /= pts_2d[..., 2]
        pts_2d[..., 1] /= pts_2d[..., 2]
        
        # FOV检查
        fov_inds = ((pts_2d[..., 0] < img_metas[0]['img_shape'][0][1]) &
                   (pts_2d[..., 0] >= 0) &
                   (pts_2d[..., 1] < img_metas[0]['img_shape'][0][0]) &
                   (pts_2d[..., 1] >= 0))
        
        # 选择第一个有效视角
        _, num_views, _, _ = pts_2d.shape
        _fov_inds = torch.cat([torch.full((fov_inds.shape[0], 1, fov_inds.shape[-1]), False, device=ref_points.device), fov_inds], dim=1)
        first_valid_view = _fov_inds.to(torch.float32).argmax(dim=1)
        first_valid_view[first_valid_view == 0] = -1
        first_valid_view[first_valid_view != -1] -= 1
        
        # 构建透视特征坐标
        batch_indices = torch.arange(batch, device=ref_points.device).unsqueeze(1).expand(-1, _num_queries)
        point_indices = torch.arange(_num_queries, device=ref_points.device).unsqueeze(0).expand(batch, -1)
        mask = first_valid_view != -1
        
        selected_pts = pts_2d[batch_indices[mask], first_valid_view[mask], point_indices[mask]]
        pts_pers = torch.full((batch, _num_queries, 2), float('nan'), device=ref_points.device)
        pts_pers[batch_indices[mask], point_indices[mask]] = selected_pts[..., [1, 0]]  # H,W
        
        # 特征图尺度调整（🔥 使用配置的尺寸，自动适应不同分辨率）
        ori_H, ori_W = img_metas[0]['img_shape'][0][:2]
        feat_H, feat_W = self.pers_h, self.pers_w  # 🔥 从配置获取，而非硬编码
        ratio = torch.tensor(feat_H / ori_H, device=ref_points.device)
        pts_pers = torch.cat([first_valid_view.unsqueeze(-1), pts_pers], dim=-1)
        pts_pers[:, :, 1:] = torch.floor(pts_pers[:, :, 1:] * ratio)
        pts_pers[pts_pers[:, :, 0] == -1] = 0.0
        
        # BEV投影（使用可配置的BEV尺寸）
        pts_bev = torch.floor((_ref_points[..., :2] + 54.0) * (self.bev_h / 108))[:, :, [1, 0]]  # y,x
        
        # 计算索引（使用可配置的尺寸）
        pts_idx = pts_bev[:, :, 0] * self.bev_w + pts_bev[:, :, 1]
        pts_pers_idx = pts_pers[:, :, 0] * self.pers_h * self.pers_w + pts_pers[:, :, 1] * self.pers_w + pts_pers[:, :, 2]
        
        return pts_bev, pts_pers, pts_idx, pts_pers_idx
    
    def generate_local_attention_masks(self, pts_idx, pts_pers_idx):
        """
        生成局部注意力掩码（LAM）
        保留MoME的精确掩码生成逻辑
        
        Args:
            pts_idx: [bs, num_queries] BEV特征图索引
            pts_pers_idx: [bs, num_queries] 透视特征图索引
            
        Returns:
            fusion_attention_mask: [bs*num_heads, num_queries, total_elements] 融合注意力掩码
        """
        batch_size, num_queries = pts_idx.shape
        
        # === Camera注意力掩码（使用可配置尺寸）===
        total_elements_cam = self.num_views * self.pers_h * self.pers_w
        H, W = self.pers_h, self.pers_w
        stride_view = self.pers_h * self.pers_w
        stride_h = self.pers_w
        window_size = self.window_sizes[0]  # camera window
        
        # 生成窗口偏移
        offsets = torch.arange(-(window_size // 2), window_size // 2 + 1, device=pts_idx.device)
        window_offsets = offsets.unsqueeze(1) * stride_h + offsets.unsqueeze(0)
        window_offsets = window_offsets.view(-1)
        
        # 应用窗口偏移
        indices = pts_pers_idx.unsqueeze(-1) + window_offsets.unsqueeze(0).unsqueeze(0)
        
        # 有效性检查
        query_rows = (pts_pers_idx % (H * W)) // W
        query_cols = pts_pers_idx % W
        index_rows = (indices % (H * W)) // W
        index_cols = indices % W
        
        valid_row = (index_rows - query_rows.unsqueeze(-1)).abs() <= window_size // 2
        valid_column = (index_cols - query_cols.unsqueeze(-1)).abs() <= window_size // 2
        valid = valid_row & valid_column
        
        indices = torch.clamp(indices, 0, total_elements_cam - 1).long()
        
        # 生成camera注意力掩码
        img_attention_mask = torch.ones(batch_size, num_queries, total_elements_cam, dtype=torch.bool, device=pts_idx.device)
        valid_indices = valid.nonzero(as_tuple=True)
        batch_indices = torch.arange(batch_size, device=pts_idx.device).long().view(-1, 1, 1)
        query_indices_range = torch.arange(num_queries, device=pts_idx.device).long().view(1, -1, 1)
        
        img_attention_mask[
            batch_indices.expand_as(indices)[valid_indices[0], valid_indices[1], valid_indices[2]],
            query_indices_range.expand_as(indices)[valid_indices[0], valid_indices[1], valid_indices[2]],
            indices[valid_indices]
        ] = False
        
        # === LiDAR注意力掩码（使用可配置尺寸）===
        total_elements_lidar = self.bev_h * self.bev_w
        row_stride = self.bev_w
        window_size = self.window_sizes[1]  # lidar window
        
        offsets = torch.arange(-(window_size // 2), window_size // 2 + 1, device=pts_idx.device)
        # 🔥 兼容PyTorch 1.9：不使用indexing参数
        y_offsets, x_offsets = torch.meshgrid(offsets, offsets)
        window_offsets = (y_offsets * row_stride + x_offsets).reshape(-1)
        
        indices = pts_idx.unsqueeze(-1) + window_offsets.unsqueeze(0).unsqueeze(0)
        valid_indices = (indices >= 0) & (indices < total_elements_lidar)
        
        # 边界检查
        query_columns = pts_idx % row_stride
        window_columns = (indices % row_stride).float() - query_columns.unsqueeze(-1).float()
        valid_indices &= (window_columns.abs() <= window_size // 2)
        
        # 生成lidar注意力掩码
        lidar_attention_mask = torch.ones(batch_size, num_queries, total_elements_lidar, dtype=torch.bool, device=pts_idx.device)
        
        batch_indices = torch.arange(batch_size, device=pts_idx.device).view(-1, 1, 1).expand_as(indices)
        query_indices_range = torch.arange(num_queries, device=pts_idx.device).view(1, -1, 1).expand_as(indices)
        
        valid_mask = valid_indices & (indices < total_elements_lidar)
        lidar_attention_mask[
            batch_indices[valid_mask],
            query_indices_range[valid_mask],
            indices[valid_mask].long()
        ] = False
        
        # 融合掩码
        fusion_attention_mask = torch.cat([lidar_attention_mask, img_attention_mask], dim=-1)
        fusion_attention_mask = fusion_attention_mask.unsqueeze(1).repeat(1, self.e_num_heads, 1, 1).flatten(0, 1)
        
        return fusion_attention_mask
    
    def forward(self, query_embed, memory, pos_embed, ref_points, img_metas, reg_branch=None):
        """
        前向传播：生成每个Query的LiDAR和Camera权重
        
        Args:
            query_embed: [num_queries, bs, embed_dims] 查询嵌入
            memory: [total_elements, bs, embed_dims] 记忆特征
            pos_embed: [total_elements, bs, embed_dims] 位置编码
            ref_points: [bs, num_queries, 3] 归一化参考点
            img_metas: 图像元数据
            reg_branch: 回归分支（可选）
            
        Returns:
            lidar_weights: [bs, num_queries] LiDAR模态权重 [0, 1]
            camera_weights: [bs, num_queries] Camera模态权重 [0, 1]
            weight_loss: 权重损失（训练时）
            projection_info: dict 投影信息，包含pts_bev, pts_pers等
        """
        # 1. 3D投影和位置映射
        pts_bev, pts_pers, pts_idx, pts_pers_idx = self.project_3d_to_features(ref_points, img_metas)
        
        # 2. 生成局部注意力掩码
        fusion_attention_mask = self.generate_local_attention_masks(pts_idx, pts_pers_idx)
        
        # 3. 编码器处理（保留MoME逻辑）
        target = torch.zeros_like(query_embed)
        target = self.encoder(
            query=target,
            key=memory,
            value=memory,
            query_pos=query_embed,
            key_pos=pos_embed,
            attn_masks=[fusion_attention_mask],
            reg_branch=reg_branch
        )
        
        # 获取最后一层输出
        if target.shape[0] == 0:
            target = target.squeeze(0).transpose(1, 0)
        else:
            target = target[-1].transpose(1, 0)  # [bs, num_queries, embed_dims]
        
        # 4. 🔥 核心修改：生成连续权重而非离散选择
        weights = self.weight_predictor(target)  # [bs, num_queries, 2]
        weights = torch.tanh(weights)  # 🔥 改用tanh：范围[-1, 1]，支持负bias（抑制）
        
        lidar_weights = weights[..., 0]   # [bs, num_queries] ∈ [-1, 1]
        camera_weights = weights[..., 1]  # [bs, num_queries] ∈ [-1, 1]
        
        # 5. 🔥 遵循老师建议：移除所有直接监督，采用端到端学习
        weight_loss = None
        # 注释掉所有权重监督损失，让检测损失自然地训练AQR模块
        # if self.training:
        #     # 所有权重监督都被移除，采用纯端到端学习
        #     weight_loss = self._compute_regularization_loss(weights)
        
        # 6. 组织投影信息
        projection_info = {
            'pts_bev': pts_bev,           # [bs, num_queries, 2] BEV坐标
            'pts_pers': pts_pers,         # [bs, num_queries, 3] 透视坐标
            'pts_idx': pts_idx,           # [bs, num_queries] BEV索引
            'pts_pers_idx': pts_pers_idx  # [bs, num_queries] 透视索引
        }
        
        return lidar_weights, camera_weights, weight_loss, projection_info
    
    # 🔥 以下函数已移除，遵循老师的端到端学习建议
    # 注释保留供参考，但不再使用
    """
    移除原因：根据老师的指导，AQR权重生成器应该完全通过最终的检测损失来学习，
    而不需要任何直接的权重监督。这更符合深度学习的端到端训练理念。
    
    原有的监督函数：
    - _compute_weight_loss_from_modalmask
    - _compute_weight_loss_from_supervision  
    - _compute_regularization_loss
    - _compute_adversarial_regularization
    
    这些都是不必要的，因为：
    1. 最终的检测损失会自然地反向传播到AQR模块
    2. 模型会隐式地学会生成最优权重
    3. 权重好坏的唯一标准是最终检测效果
    """


# 默认配置
DEFAULT_AQR_CONFIG = dict(
    type='AQRWeightGenerator',
    embed_dims=256,
    encoder_config=dict(
        type='PETRTransformerDecoder',  # 🔥 统一使用PETR
        return_intermediate=True,
        num_layers=1,
        transformerlayers=dict(
            type='PETRTransformerDecoderLayer',
            with_cp=False,
            attn_cfgs=[  # 🔥 PETR需要列表格式
                dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=4,  # 🔥 与MoME保持一致：AQR使用4头
                    dropout=0.1
                ),
            ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)
            ),
            feedforward_channels=1024,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm')  # 🔥 与MoME保持一致，只有cross_attn
        )
    ),
    window_sizes=[15, 5],  # [camera_window, lidar_window]
    use_type_embed=True,
    pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    bev_feature_shape=(180, 180),  # 🔥 默认BEV特征图尺寸（voxel_size=0.075）
    pers_feature_shape=(6, 40, 100)  # 🔥 默认透视特征图尺寸（1600x640）
)
