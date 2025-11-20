# ------------------------------------------------------------------------
# CmtAQRHead - 集成AQR权重图渲染机制的CMT检测头
# 核心功能：将AQR权重生成、权重图渲染、特征调制集成到CMT框架中
# 实现细粒度的多模态特征调制
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import warnings
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import HEADS
from ..utils.aqr_weight_generator import AQRWeightGenerator
from ..utils.weight_renderer import WeightRenderer
from ..utils.feature_modulator import FeatureModulator
from .cmt_head import CmtHead


@HEADS.register_module()
class CmtAQRHead(CmtHead):
    """
    集成AQR权重图渲染机制的CMT检测头（双模式支持）
    
    继承自CmtHead，在原有基础上增加：
    1. AQR权重生成器：为每个查询生成模态权重
    2. 权重图渲染器：将查询权重渲染到特征图空间
    3. 灵活的特征调制：支持简化模式（直接相乘）和完整模式（FeatureModulator）
    4. 保持与原CMT Transformer的完全兼容性
    5. 使用pipeline的ModalMask3D进行模态mask（无模型内部mask）
    
    特征调制模式：
    - use_simple_modulation=True: 🔥 简化模式，直接相乘，速度快
    - use_simple_modulation=False: 🛡️ 完整模式，包含残差连接、权重归一化等，更稳定
    
    Args:
        aqr_config (dict): AQR权重生成器配置
        renderer_config (dict): 权重渲染器配置
        modulator_config (dict): 特征调制器配置（仅在完整模式下使用）
        enable_aqr (bool): 是否启用AQR机制，默认True
        debug_mode (bool): 是否启用调试模式，默认False
        visualization_interval (int): 可视化间隔（仅在debug模式下），默认100
        use_simple_modulation (bool): 是否使用简化调制模式，默认False（完整模式）
        **kwargs: CmtHead的其他参数
    """
    
    def __init__(self,
                 aqr_config=None,
                 renderer_config=None,
                 modulator_config=None,
                 enable_aqr=True,
                 debug_mode=False,
                 visualization_interval=100,
                 use_simple_modulation=False,  # 🔥 新增：选择简化模式还是完整模式
                 **kwargs):
        # 先初始化父类CmtHead
        super(CmtAQRHead, self).__init__(**kwargs)
        
        self.enable_aqr = enable_aqr
        self.debug_mode = debug_mode
        self.visualization_interval = visualization_interval
        self.use_simple_modulation = use_simple_modulation  # 🔥 记录调制模式
        self._forward_count = 0
        
        if self.enable_aqr:
            # 初始化AQR组件
            self._init_aqr_components(aqr_config, renderer_config, modulator_config)
        
        # 调试信息存储
        if self.debug_mode:
            self.debug_info = {}
            self._setup_debug_hooks()
    
    def _init_aqr_components(self, aqr_config, renderer_config, modulator_config):
        """初始化AQR相关组件"""
        
        # 默认配置
        default_aqr_config = dict(
            type='AQRWeightGenerator',
            embed_dims=self.hidden_dim,
            encoder_config=dict(
                type='PETRTransformerDecoder',  # 🔥 统一使用PETR
                return_intermediate=True,
                num_layers=1,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',  # 🔥 PETR层
                    with_cp=False,
                    attn_cfgs=[  # 🔥 PETR需要列表格式
                        dict(
                            type='MultiheadAttention',
                            embed_dims=self.hidden_dim,
                            num_heads=4,  # 🔥 与MoME保持一致：AQR使用4头
                            dropout=0.1
                        ),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=self.hidden_dim,
                        feedforward_channels=self.hidden_dim * 4,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)
                    ),
                    feedforward_channels=self.hidden_dim * 4,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm')  # 🔥 与MoME保持一致，只有cross_attn
                )
            ),
            window_sizes=[15, 5],
            use_type_embed=True,
            pc_range=self.pc_range
        )
        
        default_renderer_config = dict(
            type='WeightRenderer',
            render_method='gaussian',
            gaussian_sigma=2.0,
            bev_feature_shape=(180, 180),
            pers_feature_shape=(6, 40, 100),
            normalize_weights=True
        )
        
        # 🔥 根据模式选择特征调制方式
        default_modulator_config = dict(
            type='FeatureModulator',
            modulation_type='element_wise',
            normalize_weights=True,
            residual_connection=True,
            residual_weight=0.1,
            learnable_modulation=False,
            activation='none'
        )
        
        # 合并用户配置
        if aqr_config:
            default_aqr_config.update(aqr_config)
        if renderer_config:
            default_renderer_config.update(renderer_config)
        if modulator_config:
            default_modulator_config.update(modulator_config)
        
        # 创建组件
        self.aqr_weight_generator = AQRWeightGenerator(**default_aqr_config)
        self.weight_renderer = WeightRenderer(**default_renderer_config)
        
        # 🔥 根据模式选择是否创建FeatureModulator
        if not self.use_simple_modulation:
            self.feature_modulator = FeatureModulator(**default_modulator_config)
        else:
            self.feature_modulator = None  # 使用简化模式
        
        print(f"✅ AQR components initialized successfully!")
        print(f"   - AQRWeightGenerator: {default_aqr_config['type']}")
        print(f"   - WeightRenderer: {default_renderer_config['type']} ({default_renderer_config['render_method']})")
        if self.use_simple_modulation:
            print(f"   - FeatureModulator: 🔥 Simple mode (direct multiplication)")
        else:
            print(f"   - FeatureModulator: 🛡️ Full mode ({default_modulator_config['modulation_type']}, residual={default_modulator_config['residual_connection']})")
    
    def _setup_debug_hooks(self):
        """设置调试钩子"""
        def debug_hook(module, input, output):
            if hasattr(output, 'shape'):
                self.debug_info[f'{module.__class__.__name__}_output_shape'] = output.shape
                if torch.is_tensor(output):
                    self.debug_info[f'{module.__class__.__name__}_output_stats'] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item()
                    }
        
        if hasattr(self, 'aqr_weight_generator'):
            self.aqr_weight_generator.register_forward_hook(debug_hook)
            self.weight_renderer.register_forward_hook(debug_hook)
            if hasattr(self, 'feature_modulator') and self.feature_modulator is not None:
                self.feature_modulator.register_forward_hook(debug_hook)
    
    @force_fp32(apply_to=('x', 'x_img'))
    def forward_single(self, x, x_img, img_metas):
        """
        前向传播：集成AQR权重图渲染机制
        
        Args:
            x: [bs, c, h, w] LiDAR特征图
            x_img: [bs*views, c, h, w] Camera特征图
            img_metas: 图像元数据
            
        Returns:
            ret_dicts: 检测结果字典列表
        """
        self._forward_count += 1
        ret_dicts = []
        
        # 1. 标准CMT特征预处理
        if x is not None:
            x = self.shared_conv(x)
        
        # 获取参考点
        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(
            x.shape[0] if x is not None else len(img_metas), 
            reference_points, 
            img_metas
        )
        
        # 2. 🔥 AQR权重图渲染流水线
        if self.enable_aqr and x is not None and x_img is not None:
            x, x_img = self._apply_aqr_modulation(x, x_img, reference_points, img_metas)
        
        # 3. 标准CMT位置编码和查询嵌入
        if x is not None:
            mask = x.new_zeros(x.shape[0], x.shape[2], x.shape[3])
            bev_pos_embeds = self.bev_embedding(
                self.pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim)
            )
        else:
            mask = None
            bev_pos_embeds = None
        
        if x_img is not None:
            rv_pos_embeds = self._rv_pe(x_img, img_metas)
        else:
            rv_pos_embeds = None
        
        # 查询嵌入
        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
        query_embeds = bev_query_embeds
        if rv_query_embeds is not None:
            query_embeds = query_embeds + rv_query_embeds
        
        # 4. 标准CMT Transformer处理
        outs_dec, _ = self.transformer(
            x, x_img, query_embeds,
            bev_pos_embeds, rv_pos_embeds,
            attn_masks=attn_mask
        )
        outs_dec = torch.nan_to_num(outs_dec)
        
        # 5. 标准CMT后处理和任务头
        reference = self.inverse_sigmoid(reference_points.clone())
        
        flag = 0
        for task_id, task in enumerate(self.task_heads):
            outs = task(outs_dec)
            
            # 回归分支处理
            reg_branch = None
            if 'reg_branch' in outs:
                reg_branch = outs['reg_branch']
            
            # 标准CMT的输出处理逻辑
            for key in outs.keys():
                if 'reg' in key or 'height' in key:
                    outs[key] = outs[key] + reference[..., :outs[key].shape[-1]]
                    if 'vel' in key:
                        outs[key][..., :2] = outs[key][..., :2] / self.scalar
                    else:
                        outs[key] = outs[key] / self.scalar
            
            ret_dicts.append(outs)
        
        # 6. 调试和可视化
        if self.debug_mode and self._forward_count % self.visualization_interval == 0:
            self._debug_visualization(img_metas)
        
        return ret_dicts
    
    def _apply_aqr_modulation(self, x, x_img, reference_points, img_metas):
        """
        应用AQR权重图渲染调制（简化版）
        🔥 核心改进：直接使用pipeline的ModalMask3D，特征调制简化为直接相乘
        
        Args:
            x: [bs, c, h, w] LiDAR特征图
            x_img: [bs*views, c, h, w] Camera特征图
            reference_points: [bs, num_queries, 3] 参考点
            img_metas: 图像元数据
            
        Returns:
            x_modulated: 调制后的LiDAR特征图
            x_img_modulated: 调制后的Camera特征图
        """
        # 🔥 使用pipeline的ModalMask3D，无需模型内部mask
        bs, c, h, w = x.shape
        
        # 准备融合特征和位置编码（用于AQR）
        bev_memory = x.flatten(2).transpose(1, 2)  # [bs, h*w, c]
        rv_memory = x_img.view(bs, -1, x_img.shape[1], x_img.shape[2] * x_img.shape[3])
        rv_memory = rv_memory.flatten(2).transpose(1, 2)  # [bs, views*h*w, c]
        
        # 融合memory
        memory = torch.cat([bev_memory, rv_memory], dim=1).transpose(0, 1)  # [total_elements, bs, c]
        
        # 位置编码
        bev_pos_embeds = self.bev_embedding(
            self.pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim)
        )
        rv_pos_embeds = self._rv_pe(x_img, img_metas)
        bev_pos_embeds = bev_pos_embeds.unsqueeze(1).repeat(1, bs, 1)
        rv_pos_embeds = rv_pos_embeds.view(bs, -1, self.hidden_dim).transpose(0, 1)
        
        pos_embed = torch.cat([bev_pos_embeds, rv_pos_embeds], dim=0)  # [total_elements, bs, c]
        
        # 查询嵌入
        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
        query_embed = bev_query_embeds + rv_query_embeds  # [bs, num_queries, c]
        query_embed = query_embed.transpose(0, 1)  # [num_queries, bs, c]
        
        try:
            # Step 1: AQR权重生成（端到端学习，无需权重损失）
            lidar_weights, camera_weights, _, projection_info = self.aqr_weight_generator(
                query_embed, memory, pos_embed, reference_points, img_metas
            )
            
            # Step 2: 权重图渲染
            weight_map_bev = self.weight_renderer.render_bev_weights(
                lidar_weights, projection_info['pts_bev']
            )
            weight_map_pers = self.weight_renderer.render_perspective_weights(
                camera_weights, projection_info['pts_pers']
            )
            
            # Step 3: 🔥 特征调制（支持两种模式）
            if self.use_simple_modulation:
                # 简化模式：直接相乘
                x_modulated = x * weight_map_bev.unsqueeze(1)  # [bs, c, h, w] * [bs, 1, h, w]
                x_img_modulated = x_img * weight_map_pers.view(-1, 1, weight_map_pers.shape[-2], weight_map_pers.shape[-1])
            else:
                # 完整模式：使用FeatureModulator（包含残差连接、归一化等）
                x_modulated = self.feature_modulator(x, weight_map_bev, feature_type='bev')
                x_img_modulated = self.feature_modulator(x_img, weight_map_pers, feature_type='perspective')
            
            # 存储调试信息
            if self.debug_mode:
                self.debug_info.update({
                    'lidar_weights_stats': self._compute_tensor_stats(lidar_weights),
                    'camera_weights_stats': self._compute_tensor_stats(camera_weights),
                    'weight_map_bev_stats': self._compute_tensor_stats(weight_map_bev),
                    'weight_map_pers_stats': self._compute_tensor_stats(weight_map_pers),
                    'modulation_effect_bev': self._compute_modulation_effect(x, x_modulated),
                    'modulation_effect_pers': self._compute_modulation_effect(x_img, x_img_modulated)
                })
            
            return x_modulated, x_img_modulated
            
        except Exception as e:
            warnings.warn(f"AQR modulation failed: {e}. Using original features.")
            return x, x_img
    
    # 🔥 _apply_modal_masking函数已删除，使用pipeline的ModalMask3D替代
    
    def _compute_tensor_stats(self, tensor):
        """计算张量统计信息"""
        return {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'shape': list(tensor.shape)
        }
    
    def _compute_modulation_effect(self, original, modulated):
        """计算调制效果"""
        diff = modulated - original
        return {
            'mean_change': diff.mean().item(),
            'std_change': diff.std().item(),
            'max_change': diff.abs().max().item(),
            'relative_change': (diff.abs() / (original.abs() + 1e-8)).mean().item()
        }
    
    def _debug_visualization(self, img_metas):
        """调试可视化"""
        if not hasattr(self, 'debug_info') or len(self.debug_info) == 0:
            return
        
        print(f"\n🔍 AQR Debug Info (Forward #{self._forward_count}):")
        for key, value in self.debug_info.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"     {k}: {v:.6f}")
                    else:
                        print(f"     {k}: {v}")
            else:
                print(f"   {key}: {value}")
        
        # 清空调试信息
        self.debug_info.clear()
    
    def pos2embed(self, pos, num_pos_feats=128, temperature=10000):
        """位置编码转换（复用CMT逻辑）"""
        scale = 2 * np.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        return posemb
    
    def inverse_sigmoid(self, x, eps=1e-5):
        """反sigmoid函数（复用CMT逻辑）"""
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)
    
    def get_aqr_loss(self):
        """获取AQR相关的损失（如果有的话）"""
        # 这里可以添加AQR特定的损失计算
        # 例如：权重分布的正则化损失、模态平衡损失等
        loss_dict = {}
        
        # 示例：权重平衡损失
        if hasattr(self, '_last_lidar_weights') and hasattr(self, '_last_camera_weights'):
            lidar_mean = self._last_lidar_weights.mean()
            camera_mean = self._last_camera_weights.mean()
            balance_loss = torch.abs(lidar_mean - camera_mean)
            loss_dict['aqr_balance_loss'] = balance_loss * 0.01  # 小权重
        
        return loss_dict


# 配置示例
def get_cmt_aqr_config():
    """获取CmtAQRHead的配置示例"""
    return dict(
        type='CmtAQRHead',
        # CMT Head基本配置
        in_channels=512,
        hidden_dim=256,
        num_query=900,
        # AQR特定配置
        enable_aqr=True,
        debug_mode=False,
        visualization_interval=100,
        aqr_config=dict(
            embed_dims=256,
            window_sizes=[15, 5],
            use_type_embed=True
        ),
        renderer_config=dict(
            render_method='gaussian',
            gaussian_sigma=2.0,
            normalize_weights=True
        ),
        modulator_config=dict(
            modulation_type='element_wise',
            residual_connection=True,
            residual_weight=0.1
        ),
        # 其他CMT配置...
        transformer=dict(
            type='CmtTransformer',
            # transformer配置...
        ),
        # 损失函数配置...
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0)
    )
