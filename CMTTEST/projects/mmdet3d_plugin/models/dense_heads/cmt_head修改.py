# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

from distutils.command.build import build
import enum
from turtle import down
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import xavier_init, constant_init, kaiming_init
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean, build_bbox_coder)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import NormedLinear
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.models.utils.clip_sigmoid import clip_sigmoid
from mmdet3d.models import builder
from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from einops import rearrange
import collections

from functools import reduce
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox


def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, groups, eps):
        ctx.groups = groups
        ctx.eps = eps
        N, C, L = x.size()
        x = x.view(N, groups, C // groups, L)
        mu = x.mean(2, keepdim=True)
        var = (x - mu).pow(2).mean(2, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y.view(N, C, L) + bias.view(1, C, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        groups = ctx.groups
        eps = ctx.eps

        N, C, L = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        g = g.view(N, groups, C//groups, L)
        mean_g = g.mean(dim=2, keepdim=True)
        mean_gy = (g * y).mean(dim=2, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx.view(N, C, L), (grad_output * y.view(N, C, L)).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None, None


class GroupLayerNorm1d(nn.Module):

    def __init__(self, channels, groups=1, eps=1e-6):
        super(GroupLayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.groups, self.eps)


@HEADS.register_module()
class SeparateTaskHead(BaseModule):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels,
                 heads,
                 groups=1,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(SeparateTaskHead, self).__init__(init_cfg=init_cfg)
        self.heads = heads
        self.groups = groups
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.extend([
                    nn.Conv1d(
                        c_in * groups,
                        head_conv * groups,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        groups=groups,
                        bias=False),
                    GroupLayerNorm1d(head_conv * groups, groups=groups),
                    nn.ReLU(inplace=True)
                ])
                c_in = head_conv

            conv_layers.append(
                nn.Conv1d(
                    head_conv * groups,
                    classes * groups,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    groups=groups,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            if init_cfg is None:
                self.init_cfg = dict(type='Kaiming', layer='Conv1d')

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == 'cls_logits':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [N, B, query, C].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [N, B, query, 2].
                -height (torch.Tensor): Height value with the \
                    shape of [N, B, query, 1].
                -dim (torch.Tensor): Size value with the shape \
                    of [N, B, query, 3].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [N, B, query, 2].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [N, B, query, 2].
        """
        N, B, query_num, c1 = x.shape
        x = rearrange(x, "n b q c -> b (n c) q")
        ret_dict = dict()
        
        for head in self.heads:
             head_output = self.__getattr__(head)(x)
             ret_dict[head] = rearrange(head_output, "b (n c) q -> n b q c", n=N)

        return ret_dict


@HEADS.register_module()
class CmtHead(BaseModule):

    def __init__(self,
                 in_channels,
                 num_query=900,
                 hidden_dim=128,
                 depth_num=64,
                 norm_bbox=True,
                 downsample_scale=8,
                 scalar=10,
                 noise_scale=1.0,
                 noise_trans=0.0,
                 dn_weight=1.0,
                 split=0.75,
                 train_cfg=None,
                 test_cfg=None,
                 common_heads=dict(
                     center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
                 ),
                 tasks=[
                    dict(num_class=1, class_names=['car']),
                    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                    dict(num_class=2, class_names=['bus', 'trailer']),
                    dict(num_class=1, class_names=['barrier']),
                    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
                 ],
                 transformer=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type="FocalLoss",
                     use_sigmoid=True,
                     reduction="mean",
                     gamma=2, alpha=0.25, loss_weight=1.0
                 ),
                 loss_bbox=dict(
                    type="L1Loss",
                    reduction="mean",
                    loss_weight=0.25,
                 ),
                 loss_heatmap=dict(
                     type="GaussianFocalLoss",
                     reduction="mean"
                 ),
                 separate_head=dict(
                     type='SeparateMlpHead', init_bias=-2.19, final_kernel=3),
                 # 🔥 AQR集成参数
                 enable_aqr=False,
                 aqr_config=None,
                 renderer_config=None,  # 废弃：使用attention_bias替代
                 modulator_config=None,  # 废弃：使用attention_bias替代
                 attention_bias_config=None,  # 🔥 新增：Attention Bias配置
                 use_simple_modulation=False,  # 废弃
                 debug_mode=False,
                 visualization_interval=1000,
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None
        super(CmtHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.hidden_dim = hidden_dim
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_query = num_query
        self.in_channels = in_channels
        self.depth_num = depth_num
        self.norm_bbox = norm_bbox
        self.downsample_scale = downsample_scale
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split
        
        # 🔥 AQR集成参数
        self.enable_aqr = enable_aqr
        self.debug_mode = debug_mode
        self.visualization_interval = visualization_interval
        self.use_simple_modulation = use_simple_modulation
        self._forward_count = 0
        
        # 🔥 强制打印AQR状态（诊断用）
        print(f"\n{'='*70}")
        print(f"🔧 [CmtHead.__init__] AQR Configuration Status:")
        print(f"   enable_aqr = {enable_aqr}")
        print(f"   debug_mode = {debug_mode}")
        print(f"   aqr_config is not None = {aqr_config is not None}")
        print(f"   attention_bias_config is not None = {attention_bias_config is not None}")
        print(f"{'='*70}\n")

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.fp16_enabled = False
           
        self.shared_conv = ConvModule(
            in_channels,
            hidden_dim,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d")
        )
        
        # transformer
        self.transformer = build_transformer(transformer)
        self.reference_points = nn.Embedding(num_query, 3)
        self.bev_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rv_embedding = nn.Sequential(
            nn.Linear(self.depth_num * 3, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        # task head
        self.task_heads = nn.ModuleList()
        for num_cls in self.num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(cls_logits=(num_cls, 2)))
            separate_head.update(
                in_channels=hidden_dim,
                heads=heads, num_cls=num_cls,
                groups=transformer.decoder.num_layers
            )
            self.task_heads.append(builder.build_head(separate_head))

        # assigner
        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        
        # 🔥 AQR组件初始化
        if self.enable_aqr:
            self._init_aqr_components(aqr_config, renderer_config, modulator_config, attention_bias_config)
        
        # 调试信息存储
        if self.debug_mode:
            self.debug_info = {}
            self._setup_debug_hooks()

    def init_weights(self):
        super(CmtHead, self).init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
    
    def _init_aqr_components(self, aqr_config, renderer_config, modulator_config, attention_bias_config):
        """初始化AQR相关组件"""
        from ..utils.aqr_weight_generator import AQRWeightGenerator
        from ..utils.attention_bias_generator import AttentionBiasGenerator
        # 旧组件保留以兼容旧配置
        from ..utils.weight_renderer import WeightRenderer
        from ..utils.feature_modulator import FeatureModulator
        
        # 默认配置
        # 🔥 使用PETR组件，与CMT和MoME保持一致
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
                            type='MultiFheadFlashAttention',
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
        
        # 创建组件（移除type字段）
        aqr_config_for_init = default_aqr_config.copy()
        aqr_config_for_init.pop('type', None)  # 移除type字段，因为已经知道要实例化的类
        self.aqr_weight_generator = AQRWeightGenerator(**aqr_config_for_init)
        
        renderer_config_for_init = default_renderer_config.copy()
        renderer_config_for_init.pop('type', None)  # 移除type字段
        self.weight_renderer = WeightRenderer(**renderer_config_for_init)
        
        # 🔥 根据模式选择是否创建FeatureModulator（旧方案）
        if not self.use_simple_modulation:
            modulator_config_for_init = default_modulator_config.copy()
            modulator_config_for_init.pop('type', None)  # 移除type字段
            self.feature_modulator = FeatureModulator(**modulator_config_for_init)
        else:
            self.feature_modulator = None  # 使用简化模式
        
        # 🔥 新增：AttentionBiasGenerator（新方案，推荐）
        default_attention_bias_config = dict(
            type='AttentionBiasGenerator',
            bev_feature_shape=(180, 180),
            pers_feature_shape=(6, 40, 100),
            window_size=15,  # 与LAM camera window一致
            bias_scale=2.5,  # 🔥 缩放因子初始值
            learnable_scale=True,  # 🔥 可学习的scale（推荐）
            min_scale=0.5,  # 🔥 最小scale（防止退化）
            max_scale=5.0,  # 🔥 最大scale（防止softmax饱和）
            use_local_bias=True,
            fp16=True
        )
        
        if attention_bias_config:
            default_attention_bias_config.update(attention_bias_config)
        
        bias_config_for_init = default_attention_bias_config.copy()
        bias_config_for_init.pop('type', None)
        self.attention_bias_generator = AttentionBiasGenerator(**bias_config_for_init)
        
        print(f"✅ AQR components initialized successfully!")
        print(f"   - AQRWeightGenerator: {default_aqr_config['type']}")
        print(f"   - WeightRenderer: {default_renderer_config['type']} ({default_renderer_config['render_method']}) [Legacy]")
        print(f"   - AttentionBiasGenerator: window_size={default_attention_bias_config['window_size']}, "
              f"local={default_attention_bias_config['use_local_bias']}, fp16={default_attention_bias_config['fp16']}")
        if self.use_simple_modulation:
            print(f"   - FeatureModulator: 🔥 Simple mode (direct multiplication) [Legacy]")
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

    @property
    def coords_bev(self):
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        x_size, y_size = (
            cfg['grid_size'][1] // self.downsample_scale,
            cfg['grid_size'][0] // self.downsample_scale
        )
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = (batch_x + 0.5) / x_size
        batch_y = (batch_y + 0.5) / y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
        return coord_base

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()
            
            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = sum(self.num_classes)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'known_labels_raw': known_labels_raw,
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _rv_pe(self, img_feats, img_metas):
        BN, C, H, W = img_feats.shape
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W
        coords_d = 1 + torch.arange(self.depth_num, device=img_feats[0].device).float() * (self.pc_range[3] - 1) / self.depth_num
        coords_h, coords_w, coords_d = torch.meshgrid([coords_h, coords_w, coords_d])

        coords = torch.stack([coords_w, coords_h, coords_d, coords_h.new_ones(coords_h.shape)], dim=-1)
        coords[..., :2] = coords[..., :2] * coords[..., 2:3]
        
        imgs2lidars = np.concatenate([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(coords.device)
        coords_3d = torch.einsum('hwdo, bco -> bhwdc', coords, imgs2lidars)
        coords_3d = (coords_3d[..., :3] - coords_3d.new_tensor(self.pc_range[:3])[None, None, None, :] )\
                        / (coords_3d.new_tensor(self.pc_range[3:]) - coords_3d.new_tensor(self.pc_range[:3]))[None, None, None, :]
        return self.rv_embedding(coords_3d.reshape(*coords_3d.shape[:-2], -1))

    def _bev_query_embed(self, ref_points, img_metas):
        bev_embeds = self.bev_embedding(pos2embed(ref_points, num_pos_feats=self.hidden_dim))
        return bev_embeds

    def _rv_query_embed(self, ref_points, img_metas):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        lidars2imgs = np.stack([meta['lidar2img'] for meta in img_metas])
        lidars2imgs = torch.from_numpy(lidars2imgs).float().to(ref_points.device)
        imgs2lidars = np.stack([np.linalg.inv(meta['lidar2img']) for meta in img_metas])
        imgs2lidars = torch.from_numpy(imgs2lidars).float().to(ref_points.device)

        ref_points = ref_points * (ref_points.new_tensor(self.pc_range[3:]) - ref_points.new_tensor(self.pc_range[:3])) + ref_points.new_tensor(self.pc_range[:3])
        proj_points = torch.einsum('bnd, bvcd -> bvnc', torch.cat([ref_points, ref_points.new_ones(*ref_points.shape[:-1], 1)], dim=-1), lidars2imgs)
        
        proj_points_clone = proj_points.clone()
        z_mask = proj_points_clone[..., 2:3].detach() > 0
        proj_points_clone[..., :3] = proj_points[..., :3] / (proj_points[..., 2:3].detach() + z_mask * 1e-6 - (~z_mask) * 1e-6) 
        # proj_points_clone[..., 2] = proj_points.new_ones(proj_points[..., 2].shape) 
        
        mask = (proj_points_clone[..., 0] < pad_w) & (proj_points_clone[..., 0] >= 0) & (proj_points_clone[..., 1] < pad_h) & (proj_points_clone[..., 1] >= 0)
        mask &= z_mask.squeeze(-1)

        coords_d = 1 + torch.arange(self.depth_num, device=ref_points.device).float() * (self.pc_range[3] - 1) / self.depth_num
        proj_points_clone = torch.einsum('bvnc, d -> bvndc', proj_points_clone, coords_d)
        proj_points_clone = torch.cat([proj_points_clone[..., :3], proj_points_clone.new_ones(*proj_points_clone.shape[:-1], 1)], dim=-1)
        projback_points = torch.einsum('bvndo, bvco -> bvndc', proj_points_clone, imgs2lidars)

        projback_points = (projback_points[..., :3] - projback_points.new_tensor(self.pc_range[:3])[None, None, None, :] )\
                        / (projback_points.new_tensor(self.pc_range[3:]) - projback_points.new_tensor(self.pc_range[:3]))[None, None, None, :]
        
        rv_embeds = self.rv_embedding(projback_points.reshape(*projback_points.shape[:-2], -1))
        rv_embeds = (rv_embeds * mask.unsqueeze(-1)).sum(dim=1)
        return rv_embeds

    def query_embed(self, ref_points, img_metas):
        ref_points = inverse_sigmoid(ref_points.clone()).sigmoid()
        bev_embeds = self._bev_query_embed(ref_points, img_metas)
        rv_embeds = self._rv_query_embed(ref_points, img_metas)
        return bev_embeds, rv_embeds

    def forward_single(self, x, x_img, img_metas):
        """
            x: [bs c h w]
            return List(dict(head_name: [num_dec x bs x num_query * head_dim]) ) x task_num
        """
        self._forward_count += 1
        ret_dicts = []
        x = self.shared_conv(x)
        
        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(x.shape[0], reference_points, img_metas)
        
        # 🔥 AQR Attention Bias生成（新方案）
        attention_bias = None
        if self.enable_aqr and x is not None and x_img is not None:
            attention_bias = self._generate_aqr_attention_bias(x, x_img, reference_points, img_metas)
        
        mask = x.new_zeros(x.shape[0], x.shape[2], x.shape[3])
        
        rv_pos_embeds = self._rv_pe(x_img, img_metas)
        bev_pos_embeds = self.bev_embedding(pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim))
        
        bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
        query_embeds = bev_query_embeds + rv_query_embeds

        outs_dec, _ = self.transformer(
                            x, x_img, query_embeds,
                            bev_pos_embeds, rv_pos_embeds,
                            attn_masks=attn_mask,
                            attention_bias=attention_bias  # 🔥 传递attention_bias
                        )
        outs_dec = torch.nan_to_num(outs_dec)

        reference = inverse_sigmoid(reference_points.clone())
        
        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            outs = task(outs_dec)
            center = (outs['center'] + reference[None, :, :, :2]).sigmoid()
            height = (outs['height'] + reference[None, :, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
            
            if mask_dict and mask_dict['pad_size'] > 0:
                task_mask_dict = copy.deepcopy(mask_dict)
                class_name = self.class_names[task_id]

                known_lbs_bboxes_label =  task_mask_dict['known_lbs_bboxes'][0]
                known_labels_raw = task_mask_dict['known_labels_raw']
                new_lbs_bboxes_label = known_lbs_bboxes_label.new_zeros(known_lbs_bboxes_label.shape)
                new_lbs_bboxes_label[:] = len(class_name)
                new_labels_raw = known_labels_raw.new_zeros(known_labels_raw.shape)
                new_labels_raw[:] = len(class_name)
                task_masks = [
                    torch.where(known_lbs_bboxes_label == class_name.index(i) + flag)
                    for i in class_name
                ]
                task_masks_raw = [
                    torch.where(known_labels_raw == class_name.index(i) + flag)
                    for i in class_name
                ]
                for cname, task_mask, task_mask_raw in zip(class_name, task_masks, task_masks_raw):
                    new_lbs_bboxes_label[task_mask] = class_name.index(cname)
                    new_labels_raw[task_mask_raw] = class_name.index(cname)
                task_mask_dict['known_lbs_bboxes'] = (new_lbs_bboxes_label, task_mask_dict['known_lbs_bboxes'][1])
                task_mask_dict['known_labels_raw'] = new_labels_raw
                flag += len(class_name)
                
                for key in list(outs.keys()):
                    outs['dn_' + key] = outs[key][:, :, :mask_dict['pad_size'], :]
                    outs[key] = outs[key][:, :, mask_dict['pad_size']:, :]
                outs['dn_mask_dict'] = task_mask_dict
            
            ret_dicts.append(outs)

        return ret_dicts

    def forward(self, pts_feats, img_feats=None, img_metas=None):
        """
            list([bs, c, h, w])
        """
        img_metas = [img_metas for _ in range(len(pts_feats))]
        return multi_apply(self.forward_single, pts_feats, img_feats, img_metas)
    
    def _get_targets_single(self, gt_bboxes_3d, gt_labels_3d, pred_bboxes, pred_logits):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            
            gt_bboxes_3d (Tensor):  LiDARInstance3DBoxes(num_gts, 9)
            gt_labels_3d (Tensor): Ground truth class indices (num_gts, )
            pred_bboxes (list[Tensor]): num_tasks x (num_query, 10)
            pred_logits (list[Tensor]): num_tasks x (num_query, task_classes)
        Returns:
            tuple[Tensor]: a tuple containing the following.
                - labels_tasks (list[Tensor]): num_tasks x (num_query, ).
                - label_weights_tasks (list[Tensor]): num_tasks x (num_query, ).
                - bbox_targets_tasks (list[Tensor]): num_tasks x (num_query, 9).
                - bbox_weights_tasks (list[Tensor]): num_tasks x (num_query, 10).
                - pos_inds (list[Tensor]): num_tasks x Sampled positive indices.
                - neg_inds (Tensor): num_tasks x Sampled negative indices.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)
        
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)
        
        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, dim=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        
        def task_assign(bbox_pred, logits_pred, gt_bboxes, gt_labels, num_classes):
            num_bboxes = bbox_pred.shape[0]
            assign_results = self.assigner.assign(bbox_pred, logits_pred, gt_bboxes, gt_labels)
            sampling_result = self.sampler.sample(assign_results, bbox_pred, gt_bboxes)
            pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds
            # label targets
            labels = gt_bboxes.new_full((num_bboxes, ),
                                    num_classes,
                                    dtype=torch.long)
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights = gt_bboxes.new_ones(num_bboxes)
            # bbox_targets
            code_size = gt_bboxes.shape[1]
            bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
            bbox_weights = torch.zeros_like(bbox_pred)
            bbox_weights[pos_inds] = 1.0
            
            if len(sampling_result.pos_gt_bboxes) > 0:
                bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

        labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks\
             = multi_apply(task_assign, pred_bboxes, pred_logits, task_boxes, task_classes, self.num_classes)
        
        return labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks
            
    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            pred_bboxes (list[list[Tensor]]): batch_size x num_task x [num_query, 10].
            pred_logits (list[list[Tensor]]): batch_size x num_task x [num_query, task_classes]
        Returns:
            tuple: a tuple containing the following targets.
                - task_labels_list (list(list[Tensor])): num_tasks x batch_size x (num_query, ).
                - task_labels_weight_list (list[Tensor]): num_tasks x batch_size x (num_query, )
                - task_bbox_targets_list (list[Tensor]): num_tasks x batch_size x (num_query, 9)
                - task_bbox_weights_list (list[Tensor]): num_tasks x batch_size x (num_query, 10)
                - num_total_pos_tasks (list[int]): num_tasks x Number of positive samples
                - num_total_neg_tasks (list[int]): num_tasks x Number of negative samples.
        """
        (labels_list, labels_weight_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits
        )
        task_num = len(labels_list[0])
        num_total_pos_tasks, num_total_neg_tasks = [], []
        task_labels_list, task_labels_weight_list, task_bbox_targets_list, \
            task_bbox_weights_list = [], [], [], []

        for task_id in range(task_num):
            num_total_pos_task = sum((inds[task_id].numel() for inds in pos_inds_list))
            num_total_neg_task = sum((inds[task_id].numel() for inds in neg_inds_list))
            num_total_pos_tasks.append(num_total_pos_task)
            num_total_neg_tasks.append(num_total_neg_task)
            task_labels_list.append([labels_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_labels_weight_list.append([labels_weight_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_targets_list.append([bbox_targets_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_weights_list.append([bbox_weights_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
        
        return (task_labels_list, task_labels_weight_list, task_bbox_targets_list,
                task_bbox_weights_list, num_total_pos_tasks, num_total_neg_tasks)
        
    def _loss_single_task(self,
                          pred_bboxes,
                          pred_logits,
                          labels_list,
                          labels_weights_list,
                          bbox_targets_list,
                          bbox_weights_list,
                          num_total_pos,
                          num_total_neg):
        """"Compute loss for single task.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            pred_bboxes (Tensor): (batch_size, num_query, 10)
            pred_logits (Tensor): (batch_size, num_query, task_classes)
            labels_list (list[Tensor]): batch_size x (num_query, )
            labels_weights_list (list[Tensor]): batch_size x (num_query, )
            bbox_targets_list(list[Tensor]): batch_size x (num_query, 9)
            bbox_weights_list(list[Tensor]): batch_size x (num_query, 10)
            num_total_pos: int
            num_total_neg: int
        Returns:
            loss_cls
            loss_bbox 
        """
        labels = torch.cat(labels_list, dim=0)
        labels_weights = torch.cat(labels_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)
        
        pred_bboxes_flatten = pred_bboxes.flatten(0, 1)
        pred_logits_flatten = pred_logits.flatten(0, 1)
        
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0.1
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits_flatten, labels, labels_weights, avg_factor=cls_avg_factor
        )

        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]

        loss_bbox = self.loss_bbox(
            pred_bboxes_flatten[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox) 
        return loss_cls, loss_bbox

    def loss_single(self,
                    pred_bboxes,
                    pred_logits,
                    gt_bboxes_3d,
                    gt_labels_3d):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            pred_bboxes (list[Tensor]): num_tasks x [bs, num_query, 10].
            pred_logits (list(Tensor]): num_tasks x [bs, num_query, task_classes]
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        batch_size = pred_bboxes[0].shape[0]
        pred_bboxes_list, pred_logits_list = [], []
        for idx in range(batch_size):
            pred_bboxes_list.append([task_pred_bbox[idx] for task_pred_bbox in pred_bboxes])
            pred_logits_list.append([task_pred_logits[idx] for task_pred_logits in pred_logits])
        cls_reg_targets = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, pred_bboxes_list, pred_logits_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._loss_single_task, 
            pred_bboxes,
            pred_logits,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg
        )

        return sum(loss_cls_tasks), sum(loss_bbox_tasks)
    
    def _dn_loss_single_task(self,
                             pred_bboxes,
                             pred_logits,
                             mask_dict):
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()

        # 🔥 修复设备不匹配问题
        # 确保 known_indice 和 batch_idx 在同一设备上
        if known_indice.device != batch_idx.device:
            known_indice = known_indice.to(batch_idx.device)
        if map_known_indice.device != batch_idx.device:
            map_known_indice = map_known_indice.to(batch_idx.device)

        bid = batch_idx[known_indice]
        known_labels_raw = mask_dict['known_labels_raw']

                # 🔥 确保 known_labels_raw 也在正确设备上
        if known_labels_raw.device != batch_idx.device:
            known_labels_raw = known_labels_raw.to(batch_idx.device)
        
        pred_logits = pred_logits[(bid, map_known_indice)]
        pred_bboxes = pred_bboxes[(bid, map_known_indice)]
        num_tgt = known_indice.numel()

        # filter task bbox
        task_mask = known_labels_raw != pred_logits.shape[-1]
        task_mask_sum = task_mask.sum()
        
        if task_mask_sum > 0:
            # pred_logits = pred_logits[task_mask]
            # known_labels = known_labels[task_mask]
            pred_bboxes = pred_bboxes[task_mask]
            known_bboxs = known_bboxs[task_mask]

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split  * self.split
        
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_tgt = loss_cls.new_tensor([num_tgt])
        num_tgt = torch.clamp(reduce_mean(num_tgt), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = torch.ones_like(pred_bboxes)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
        # bbox_weights[:, 6:8] = 0
        loss_bbox = self.loss_bbox(
                pred_bboxes[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_tgt)
 
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        if task_mask_sum == 0:
            # loss_cls = loss_cls * 0.0
            loss_bbox = loss_bbox * 0.0

        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    def dn_loss_single(self,
                       pred_bboxes,
                       pred_logits,
                       dn_mask_dict):
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._dn_loss_single_task, pred_bboxes, pred_logits, dn_mask_dict
        )
        return sum(loss_cls_tasks), sum(loss_bbox_tasks)
        
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """"Loss function.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            preds_dicts(tuple[list[dict]]): nb_tasks x num_lvl
                center: (num_dec, batch_size, num_query, 2)
                height: (num_dec, batch_size, num_query, 1)
                dim: (num_dec, batch_size, num_query, 3)
                rot: (num_dec, batch_size, num_query, 2)
                vel: (num_dec, batch_size, num_query, 2)
                cls_logits: (num_dec, batch_size, num_query, task_classes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_decoder = preds_dicts[0][0]['center'].shape[0]
        all_pred_bboxes, all_pred_logits = collections.defaultdict(list), collections.defaultdict(list)

        for task_id, preds_dict in enumerate(preds_dicts, 0):
            for dec_id in range(num_decoder):
                pred_bbox = torch.cat(
                    (preds_dict[0]['center'][dec_id], preds_dict[0]['height'][dec_id],
                    preds_dict[0]['dim'][dec_id], preds_dict[0]['rot'][dec_id],
                    preds_dict[0]['vel'][dec_id]),
                    dim=-1
                )
                all_pred_bboxes[dec_id].append(pred_bbox)
                all_pred_logits[dec_id].append(preds_dict[0]['cls_logits'][dec_id])
        all_pred_bboxes = [all_pred_bboxes[idx] for idx in range(num_decoder)]
        all_pred_logits = [all_pred_logits[idx] for idx in range(num_decoder)]

        loss_cls, loss_bbox = multi_apply(
            self.loss_single, all_pred_bboxes, all_pred_logits,
            [gt_bboxes_3d for _ in range(num_decoder)],
            [gt_labels_3d for _ in range(num_decoder)], 
        )

        loss_dict = dict()
        loss_dict['loss_cls'] = loss_cls[-1]
        loss_dict['loss_bbox'] = loss_bbox[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(loss_cls[:-1],
                                           loss_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        
        dn_pred_bboxes, dn_pred_logits = collections.defaultdict(list), collections.defaultdict(list)
        dn_mask_dicts = collections.defaultdict(list)
        for task_id, preds_dict in enumerate(preds_dicts, 0):
            for dec_id in range(num_decoder):
                pred_bbox = torch.cat(
                    (preds_dict[0]['dn_center'][dec_id], preds_dict[0]['dn_height'][dec_id],
                    preds_dict[0]['dn_dim'][dec_id], preds_dict[0]['dn_rot'][dec_id],
                    preds_dict[0]['dn_vel'][dec_id]),
                    dim=-1
                )
                dn_pred_bboxes[dec_id].append(pred_bbox)
                dn_pred_logits[dec_id].append(preds_dict[0]['dn_cls_logits'][dec_id])
                dn_mask_dicts[dec_id].append(preds_dict[0]['dn_mask_dict'])
        dn_pred_bboxes = [dn_pred_bboxes[idx] for idx in range(num_decoder)]
        dn_pred_logits = [dn_pred_logits[idx] for idx in range(num_decoder)]
        dn_mask_dicts = [dn_mask_dicts[idx] for idx in range(num_decoder)]
        dn_loss_cls, dn_loss_bbox = multi_apply(
            self.dn_loss_single, dn_pred_bboxes, dn_pred_logits, dn_mask_dicts
        )

        loss_dict['dn_loss_cls'] = dn_loss_cls[-1]
        loss_dict['dn_loss_bbox'] = dn_loss_bbox[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(dn_loss_cls[:-1],
                                           dn_loss_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
            num_dec_layer += 1


        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
    
    def _apply_aqr_modulation(self, x, x_img, reference_points, img_metas):
        """
        应用AQR权重图渲染调制
        🔥 核心改进：直接使用pipeline的ModalMask3D，特征调制支持双模式
        
        Args:
            x: [bs, c, h, w] LiDAR特征图
            x_img: [bs*views, c, h, w] Camera特征图
            reference_points: [bs, num_queries, 3] 参考点
            img_metas: 图像元数据
            
        Returns:
            x_modulated: 调制后的LiDAR特征图
            x_img_modulated: 调制后的Camera特征图
        """
        import warnings
        # 🔥 使用pipeline的ModalMask3D，无需模型内部mask
        bs, c, h, w = x.shape
        
        # 准备融合特征和位置编码（用于AQR）
        bev_memory = x.flatten(2).transpose(1, 2)  # [bs, h*w, c]
        
        # 🔥 修复：x_img的形状是[bs*views, c, h, w]，需要正确reshape
        BN, C, H, W = x_img.shape
        num_views = BN // bs  # 通常是6
        rv_memory = x_img.view(bs, num_views, C, H, W)  # [bs, views, c, h, w]
        rv_memory = rv_memory.permute(0, 1, 3, 4, 2)  # [bs, views, h, w, c]
        rv_memory = rv_memory.flatten(1, 3)  # [bs, views*h*w, c]
        
        # 融合memory
        memory = torch.cat([bev_memory, rv_memory], dim=1).transpose(0, 1)  # [total_elements, bs, c]
        
        # 位置编码
        bev_pos_embeds = self.bev_embedding(
            pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim)
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
                
            # 🔥 调试模式：保存权重图和调制后的特征用于可视化
            if self.debug_mode and self._forward_count % self.visualization_interval == 0:
                import os
                debug_dir = 'aqr_debug_weights'
                os.makedirs(debug_dir, exist_ok=True)
                
                # 保存权重图、调制后的特征和元数据
                save_data = {
                    'iteration': self._forward_count,
                    # 权重相关
                    'weight_map_bev': weight_map_bev.detach().cpu(),  # [bs, 180, 180]
                    'weight_map_pers': weight_map_pers.detach().cpu(),  # [bs, 6*40*100]
                    'lidar_weights': lidar_weights.detach().cpu(),  # [bs, num_queries]
                    'camera_weights': camera_weights.detach().cpu(),  # [bs, num_queries]
                    'pts_bev': projection_info['pts_bev'].detach().cpu(),  # 投影坐标
                    # 🔥 新增：调制后的特征
                    'modulated_bev_features': x_modulated.detach().cpu(),  # [bs, c, 180, 180]
                    'modulated_pers_features': x_img_modulated.detach().cpu(),  # [bs*6, c, 40, 100]
                    # 🔥 新增：原始特征（用于对比）
                    'original_bev_features': x.detach().cpu(),  # [bs, c, 180, 180]
                    'original_pers_features': x_img.detach().cpu(),  # [bs*6, c, 40, 100]
                    # 元数据
                    'img_metas': img_metas,  # 包含GT信息
                }
                
                save_path = os.path.join(debug_dir, f'weights_iter_{self._forward_count}.pth')
                torch.save(save_data, save_path)
                print(f"🐾 Saved AQR weights and modulated features to: {save_path}")
                
                self._debug_visualization(img_metas)
            
            return x_modulated, x_img_modulated
            
        except Exception as e:
            warnings.warn(f"AQR modulation failed: {e}. Using original features.")
            return x, x_img
    
    def _generate_aqr_attention_bias(self, x, x_img, reference_points, img_metas):
        """
        🔥 生成AQR Attention Bias（新方案）
        
        Args:
            x: [bs, c, h, w] LiDAR特征图
            x_img: [bs*views, c, h, w] Camera特征图
            reference_points: [bs, num_queries, 3] 参考点
            img_metas: 图像元数据
            
        Returns:
            attention_bias: [bs, num_queries, num_features] Attention bias矩阵
        """
        import warnings
        
        # 🔥 入口打印：确保这个函数被调用
        if self._forward_count == 1:
            print(f"\n🎯 [AQR] _generate_aqr_attention_bias() called for the first time!")
            print(f"   Input shapes: x={x.shape}, x_img={x_img.shape}, ref_points={reference_points.shape}")
        
        try:
            bs, c, h, w = x.shape
            
            # Step 1: 准备融合特征和位置编码（用于AQR）
            bev_memory = x.flatten(2).transpose(1, 2)  # [bs, h*w, c]
            
            # 处理Camera特征
            BN, C, H, W = x_img.shape
            num_views = BN // bs
            rv_memory = x_img.view(bs, num_views, C, H, W)  # [bs, views, c, h, w]
            rv_memory = rv_memory.permute(0, 1, 3, 4, 2)  # [bs, views, h, w, c]
            rv_memory = rv_memory.flatten(1, 3)  # [bs, views*h*w, c]
            
            # 融合memory
            memory = torch.cat([bev_memory, rv_memory], dim=1).transpose(0, 1)  # [total_elements, bs, c]
            
            # Step 2: 位置编码
            bev_pos_embeds = self.bev_embedding(
                pos2embed(self.coords_bev.to(x.device), num_pos_feats=self.hidden_dim)
            )
            rv_pos_embeds = self._rv_pe(x_img, img_metas)
            bev_pos_embeds = bev_pos_embeds.unsqueeze(1).repeat(1, bs, 1)
            rv_pos_embeds = rv_pos_embeds.view(bs, -1, self.hidden_dim).transpose(0, 1)
            
            pos_embed = torch.cat([bev_pos_embeds, rv_pos_embeds], dim=0)  # [total_elements, bs, c]
            
            # Step 3: 查询嵌入
            bev_query_embeds, rv_query_embeds = self.query_embed(reference_points, img_metas)
            query_embed = bev_query_embeds + rv_query_embeds  # [bs, num_queries, c]
            query_embed = query_embed.transpose(0, 1)  # [num_queries, bs, c]
            
            # Step 4: AQR权重生成
            lidar_weights, camera_weights, _, projection_info = self.aqr_weight_generator(
                query_embed, memory, pos_embed, reference_points, img_metas
            )
            # lidar_weights: [bs, num_queries]
            # camera_weights: [bs, num_queries]
            
            # Step 5: 🔥 使用AttentionBiasGenerator生成局部bias
            attention_bias = self.attention_bias_generator(
                lidar_weights=lidar_weights,
                camera_weights=camera_weights,
                pts_bev_indices=projection_info['pts_idx'],      # 🔥 [bs, num_queries] BEV 1D索引
                pts_pers_indices=projection_info['pts_pers']     # 🔥 [bs, num_queries, 3] 透视3D坐标 (view, h, w)
            )
            # attention_bias: [bs, num_queries, num_features=56400]
            
            # Step 6: 存储调试信息
            if self.debug_mode:
                self.debug_info.update({
                    'aqr_lidar_weights_stats': self._compute_tensor_stats(lidar_weights),
                    'aqr_camera_weights_stats': self._compute_tensor_stats(camera_weights),
                    'attention_bias_stats': self._compute_tensor_stats(attention_bias),
                })
                
                # 打印调试信息
                if self._forward_count % self.visualization_interval == 0:
                    # 🔥 获取当前的 bias_scale（如果是可学习的）
                    if hasattr(self.attention_bias_generator, 'bias_scale'):
                        if self.attention_bias_generator.learnable_scale:
                            current_scale = self.attention_bias_generator.bias_scale.item()
                            scale_info = f" | Bias Scale: {current_scale:.4f}"
                        else:
                            scale_info = f" | Bias Scale: {self.attention_bias_generator.bias_scale.item():.4f} (fixed)"
                    else:
                        scale_info = ""

                    print(f"\n🔥 AQR Attention Bias Debug (Iteration {self._forward_count}):")
                    print(f"   LiDAR Weights: mean={lidar_weights.mean():.4f}, std={lidar_weights.std():.4f}, "
                          f"min={lidar_weights.min():.4f}, max={lidar_weights.max():.4f}")
                    print(f"   Camera Weights: mean={camera_weights.mean():.4f}, std={camera_weights.std():.4f}, "
                          f"min={camera_weights.min():.4f}, max={camera_weights.max():.4f}")
                    print(f"   Attention Bias: mean={attention_bias.mean():.4f}, std={attention_bias.std():.4f}, "
                          f"min={attention_bias.min():.4f}, max={attention_bias.max():.4f}")
                    
                    # 🔥 可视化bias到相机视图
                    try:
                        self._visualize_bias_on_camera_views(
                            attention_bias=attention_bias,
                            pts_pers_indices=projection_info['pts_pers'],
                            camera_weights=camera_weights,
                            img_metas=img_metas,
                            iteration=self._forward_count
                        )
                    except Exception as e:
                        print(f"⚠️  Bias visualization failed: {e}")
                        import traceback
                        traceback.print_exc()
            
            return attention_bias
            
        except Exception as e:
            # 🔥 强制打印错误信息，不使用warnings（warnings可能被过滤）
            print(f"\n{'='*70}")
            print(f"❌ [AQR ERROR] Attention Bias Generation Failed!")
            print(f"   Error Type: {type(e).__name__}")
            print(f"   Error Message: {str(e)}")
            print(f"   Using None (no bias) for this forward pass.")
            print(f"{'='*70}\n")
            import traceback
            traceback.print_exc()
            return None
    
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
    
    def _visualize_bias_on_camera_views(self, attention_bias, pts_pers_indices, camera_weights, 
                                        img_metas, iteration, batch_idx=0):
        """
        🔥 在相机视图上可视化AQR生成的attention bias
        
        Args:
            attention_bias: [bs, num_queries, total_features] Attention bias矩阵
            pts_pers_indices: [bs, num_queries, 3] 透视特征图位置索引 (view, h, w)
            camera_weights: [bs, num_queries] Camera权重
            img_metas: 图像元数据
            iteration: 当前迭代次数
            batch_idx: 要可视化的batch索引
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from PIL import Image
        import mmcv
        
        # 1. 获取特征图尺寸
        pers_h = self.attention_bias_generator.pers_h  # 40
        pers_w = self.attention_bias_generator.pers_w  # 100
        num_views = self.attention_bias_generator.num_views  # 6
        bev_h = self.attention_bias_generator.bev_h  # 180
        bev_w = self.attention_bias_generator.bev_w  # 180
        
        # 2. 提取Camera部分的bias（后 num_views*pers_h*pers_w 个特征）
        bev_features = bev_h * bev_w
        camera_features = num_views * pers_h * pers_w
        camera_bias = attention_bias[batch_idx, :, bev_features:bev_features + camera_features]  # [num_queries, camera_features]
        
        # 3. 将bias聚合到每个视角的特征图空间
        # 初始化聚合图：对每个特征图位置，聚合所有queries在该位置的bias值
        bias_maps = torch.zeros(num_views, pers_h, pers_w, device=camera_bias.device, dtype=camera_bias.dtype)
        
        # 获取当前batch的透视索引
        pers_indices = pts_pers_indices[batch_idx]  # [num_queries, 3]
        
        # 对每个query，将其bias值添加到对应的特征图位置
        # 每个query的bias分布在一个窗口内，我们需要聚合整个窗口
        for q_idx in range(camera_bias.shape[0]):
            view_idx = int(pers_indices[q_idx, 0].item())
            
            # 检查索引有效性
            if view_idx < 0 or view_idx >= num_views:
                continue
            
            # 获取该query的所有bias值（窗口内的）
            query_camera_bias = camera_bias[q_idx]  # [camera_features]
            
            # 计算该query在特征图中的1D起始索引
            view_start_idx = view_idx * pers_h * pers_w
            view_end_idx = (view_idx + 1) * pers_h * pers_w
            
            # 提取该视角的bias
            view_bias = query_camera_bias[view_start_idx:view_end_idx]  # [pers_h*pers_w]
            view_bias_2d = view_bias.view(pers_h, pers_w)  # [pers_h, pers_w]
            
            # 聚合：保留原始bias值（带正负号），使用最大值聚合
            # 对于正bias和负bias分别处理，保留符号信息
            # 使用绝对值比较，但保留原始符号
            abs_bias_2d = view_bias_2d.abs()
            abs_current = bias_maps[view_idx].abs()
            # 如果新的绝对值更大，则更新（保留原始符号）
            mask = abs_bias_2d > abs_current
            bias_maps[view_idx] = torch.where(mask, view_bias_2d, bias_maps[view_idx])
        
        # 4. 转换为numpy并移动到CPU
        bias_maps_np = bias_maps.detach().cpu().numpy()  # [num_views, pers_h, pers_w]
        
        # 5. 创建保存目录
        vis_dir = 'aqr_bias_visualization'
        os.makedirs(vis_dir, exist_ok=True)
        
        # 6. 获取原始图像路径（从img_metas）
        view_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 
                     'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        
        if batch_idx < len(img_metas):
            img_meta = img_metas[batch_idx]
            img_filenames = img_meta.get('filename', [])
            
            # 7. 为每个视角创建可视化
            fig, axes = plt.subplots(num_views, 2, figsize=(20, 6*num_views))
            if num_views == 1:
                axes = axes.reshape(1, -1)
            
            for view_idx in range(num_views):
                # 7.1 获取原始图像
                if view_idx < len(img_filenames) and os.path.exists(img_filenames[view_idx]):
                    try:
                        img = mmcv.imread(img_filenames[view_idx])
                        img_rgb = mmcv.bgr2rgb(img)
                        img_h, img_w = img_rgb.shape[:2]
                    except:
                        img_rgb = None
                        img_h, img_w = None, None
                else:
                    img_rgb = None
                    img_h, img_w = None, None
                
                # 7.2 获取bias map（确保是float类型的numpy数组）
                bias_map = bias_maps_np[view_idx].astype(np.float32)  # [pers_h, pers_w]
                
                # 7.3 左图：原始图像（如果有）+ bias热力图叠加
                ax_left = axes[view_idx, 0]
                if img_rgb is not None and img_h is not None and img_w is not None:
                    try:
                        # ====== 美化版 bias overlay（替换原有 overlay 代码） ======

                        # ====== Ensure bias_map_upsampled exists (upsample bias_map to image size) ======
                        import numpy as np
                        import cv2
                        try:
                            # If bias_map_upsampled already exists, keep it
                            bias_map_upsampled
                        except NameError:
                            try:
                                # Prefer scipy's zoom for floating point upsampling (keeps subpixel)
                                from scipy.ndimage import zoom as _zoom
                                zoom_factors = (float(img_h) / float(pers_h), float(img_w) / float(pers_w))
                                bias_map_upsampled = _zoom(bias_map, zoom_factors, order=1).astype(np.float32)
                            except Exception:
                                # Fallback to OpenCV resize
                                try:
                                    bias_map_upsampled = cv2.resize(bias_map.astype(np.float32), (int(img_w), int(img_h)), interpolation=cv2.INTER_LINEAR)
                                except Exception:
                                    # Last-resort: tile or trim the bias_map to match size
                                    bh, bw = bias_map.shape
                                    bias_map_upsampled = np.array(bias_map, dtype=np.float32)
                                    bias_map_upsampled = np.tile(bias_map_upsampled, (int(np.ceil(img_h/bh)), int(np.ceil(img_w/bw))))[:int(img_h), :int(img_w)]
                        # ========================================================================
                        import cv2
                        from scipy.ndimage import gaussian_filter

                        # bias_map_upsampled: 已被上采样到 img 大小，float32，范围可正负
                        # img_rgb: 原始 RGB 图像，uint8

                        # 参数（可调）
                        SIGMA_BLUR = 6          # 对 alpha map 做高斯模糊（越大越柔和）
                        ALPHA_MAX = 0.55        # overlay 最大透明度
                        SHOW_THRESHOLD = 0.05   # 只显示超过该绝对值的bias（以归一化后数值为准）
                        GLOW_INTENSITY = 0.9    # 边缘 glow 强度（0-1），0 为不添加
                        GLOW_BLUR = 18          # glow 的模糊半径（越大越发散）
                        POS_COLOR_START = np.array([180, 60, 60]) / 255.0   # 深红（RGB 归一化）
                        POS_COLOR_END   = np.array([1.0, 0.6, 0.6])        # 浅粉红（更亮）
                        NEG_COLOR_START = np.array([60, 90, 180]) / 255.0  # 深蓝
                        NEG_COLOR_END   = np.array([0.6, 0.8, 1.0])       # 浅天蓝

                        # 1) 归一化绝对值（按全图最大绝对值）
                        bias = bias_map_upsampled.astype(np.float32)
                        abs_max = max(1e-6, np.max(np.abs(bias)))
                        bias_norm = bias / (abs_max + 1e-8)   # 归一化到 [-1,1]

                        # 2) 分离正/负并做平滑（高斯模糊使区域更连贯）
                        pos = np.clip(bias_norm, 0.0, 1.0)
                        neg = np.clip(-bias_norm, 0.0, 1.0)

                        # 对 pos/neg 做高斯平滑（减少颗粒、增强连贯感）
                        pos_smooth = gaussian_filter(pos, sigma=SIGMA_BLUR)
                        neg_smooth = gaussian_filter(neg, sigma=SIGMA_BLUR)

                        # 3) 构造 color maps（从深到浅的渐变颜色）
                        def color_lerp(start, end, t):
                            return (1.0 - t[..., None]) * start[None, None, :] + t[..., None] * end[None, None, :]

                        pos_color_map = color_lerp(POS_COLOR_START, POS_COLOR_END, pos_smooth)
                        neg_color_map = color_lerp(NEG_COLOR_START, NEG_COLOR_END, neg_smooth)

                        # 4) 合成 heatmap（优先正 bias，如果同时存在则叠加，权重可控）
                        heatmap = pos_color_map * pos_smooth[..., None] + neg_color_map * neg_smooth[..., None]

                        # 5) 计算 alpha map：取 pos/neg 最大，做平滑与阈值
                        alpha_map = np.maximum(pos_smooth, neg_smooth)
                        alpha_map[alpha_map < SHOW_THRESHOLD] = 0.0
                        alpha_map = np.clip(alpha_map, 0.0, 1.0) * ALPHA_MAX
                        alpha_map = gaussian_filter(alpha_map, sigma=SIGMA_BLUR/2.0)

                        # 6) Glow（可选）：在 pos 区域添加红色发光（仅视觉效果）
                        glow = np.zeros_like(alpha_map)
                        if GLOW_INTENSITY > 0:
                            glow_mask = (pos_smooth > max(SHOW_THRESHOLD * 0.5, 0.01)).astype(np.float32) * pos_smooth
                            glow = gaussian_filter(glow_mask, sigma=GLOW_BLUR) * GLOW_INTENSITY
                            glow = np.clip(glow, 0.0, 1.0)

                        # 7) 将 img_rgb 转为 float 0-1，按 alpha 混合
                        img_f = img_rgb.astype(np.float32) / 255.0
                        overlay = img_f.copy()

                        # 主体混合（heatmap 覆盖）
                        overlay = (1.0 - alpha_map[..., None]) * overlay + (alpha_map[..., None]) * heatmap

                        # 叠加 glow（只在 pos 区域，并以红色增加亮度）
                        if np.any(glow > 0):
                            glow_color = POS_COLOR_START  # glow 使用深红基色
                            glow_rgb = glow[..., None] * glow_color[None, None, :]
                            overlay = np.clip(overlay + glow_rgb * 0.25, 0.0, 1.0)

                        # 8) 转回 uint8 并显示
                        overlay_uint8 = (overlay * 255).astype(np.uint8)
                        ax_left.imshow(overlay_uint8)
                        ax_left.set_title(f'{view_names[view_idx]} - Bias Overlay (Iter {iteration})')

                    except Exception as e:
                        # 如果上采样失败，只显示原始图像
                        print(f"⚠️  Failed to overlay bias on {view_names[view_idx]}: {e}")
                        ax_left.imshow(img_rgb)
                        ax_left.set_title(f'{view_names[view_idx]} - Original Image (Iter {iteration})')
                else:
                    # 如果没有原始图像，只显示bias map
                    # 🔥 只显示增强的前50%和抑制的前50%，中间区域设为中性色
                    bias_flat = bias_map.flatten()
                    
                    # 分离正bias和负bias
                    positive_bias = bias_flat[bias_flat > 0]
                    negative_bias = bias_flat[bias_flat < 0]
                    
                    # 计算50%分位数阈值
                    if len(positive_bias) > 0:
                        positive_threshold = np.percentile(positive_bias, 50)
                    else:
                        positive_threshold = 0
                    
                    if len(negative_bias) > 0:
                        negative_threshold = np.percentile(negative_bias, 50)
                    else:
                        negative_threshold = 0
                    
                    # 创建mask：只显示前50%的增强和后50%的抑制
                    mask = (bias_map > positive_threshold) | (bias_map < negative_threshold)
                    
                    # 归一化bias值用于colormap
                    bias_max = bias_map.max()
                    bias_min = bias_map.min()
                    bias_abs_max = max(abs(bias_max), abs(bias_min))
                    
                    if bias_abs_max > 1e-6:
                        bias_normalized = bias_map / (bias_abs_max + 1e-8)
                        bias_normalized = (bias_normalized + 1.0) / 2.0
                    else:
                        bias_normalized = np.ones_like(bias_map) * 0.5
                    
                    # 中间区域设为中性色（0.5，对应黄色）
                    bias_normalized[~mask] = 0.5
                    
                    im = ax_left.imshow(bias_normalized, cmap='RdYlBu', interpolation='bilinear', vmin=0, vmax=1)
                    ax_left.set_title(f'{view_names[view_idx]} - Bias Map (Top 50% Red=+, Bottom 50% Blue=-, Iter {iteration})')
                    cbar = plt.colorbar(im, ax=ax_left)
                    # 设置colorbar标签
                    cbar.set_ticks([0.0, 0.5, 1.0])
                    cbar.set_ticklabels([f'Bottom 50% Negative\n(≤{negative_threshold:.2f})', 'Neutral (Hidden)', f'Top 50% Positive\n(≥{positive_threshold:.2f})'])
                
                ax_left.axis('off')
                
                # 7.4 右图：bias统计和Camera权重
                ax_right = axes[view_idx, 1]
                ax_right.axis('off')
                
                # 显示统计信息
                stats_text = f"View: {view_names[view_idx]}\n"
                stats_text += f"Bias Map Shape: {bias_map.shape}\n"
                stats_text += f"Bias Range: [{bias_map.min():.4f}, {bias_map.max():.4f}]\n"
                stats_text += f"Bias Mean: {bias_map.mean():.4f}\n"
                stats_text += f"Bias Std: {bias_map.std():.4f}\n\n"
                
                # 计算该视角的Camera权重统计
                view_mask = (pers_indices[:, 0] == view_idx)
                if view_mask.sum() > 0:
                    view_camera_weights = camera_weights[batch_idx][view_mask].detach().cpu().numpy()
                    stats_text += f"Camera Weights (View {view_idx}):\n"
                    stats_text += f"  Mean: {view_camera_weights.mean():.4f}\n"
                    stats_text += f"  Std: {view_camera_weights.std():.4f}\n"
                    stats_text += f"  Range: [{view_camera_weights.min():.4f}, {view_camera_weights.max():.4f}]\n"
                    stats_text += f"  Queries: {view_mask.sum().item()}\n"
                
                ax_right.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 8. 保存图像
            save_path = os.path.join(vis_dir, f'bias_visualization_iter_{iteration}_batch_{batch_idx}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📸 Saved bias visualization to: {save_path}")
        else:
            print(f"⚠️  Batch index {batch_idx} out of range for img_metas")

