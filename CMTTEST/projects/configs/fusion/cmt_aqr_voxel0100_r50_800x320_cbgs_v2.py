# 基于AQR权重图渲染机制的CMT配置文件 (800×320分辨率)
# 完整重写pts_bbox_head配置，确保AQR参数被正确加载

# 继承原始配置
_base_ = ['./cmt_voxel0100_r50_800x320_cbgs.py']

# 定义点云范围和体素大小（从base继承）
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.1, 0.1, 0.2]
out_size_factor = 8

# 🔥 完整重写model配置，确保AQR参数生效
model = dict(
    pts_bbox_head=dict(
        # ========== 基础参数（从base继承） ==========
        type='CmtHead',
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        tasks=[
            dict(num_class=10, class_names=[
                'car', 'truck', 'construction_vehicle',
                'bus', 'trailer', 'barrier',
                'motorcycle', 'bicycle',
                'pedestrian', 'traffic_cone'
            ]),
        ],
        bbox_coder=dict(
            type='MultiTaskBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10
        ), 
        separate_head=dict(
            type='SeparateTaskHead', 
            init_bias=-2.19, 
            final_kernel=1
        ),
        transformer=dict(
            type='CmtTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1
                        ),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1
                        ),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=1024,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                )
            )
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        
        # ========== 🔥 AQR特定配置（新增） ==========
        enable_aqr=True,               # ✅ 启用AQR机制
        debug_mode=True,              # 调试模式
        visualization_interval=100,    # 可视化间隔
        use_simple_modulation=False,   # 使用Attention Bias方案
        
        # AQR权重生成器配置
        aqr_config=dict(
            embed_dims=256,
            window_sizes=[8, 5],      # [camera_window, lidar_window]
            use_type_embed=True,
            bev_feature_shape=(128, 128),
            pers_feature_shape=(6, 20, 50),
            encoder_config=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=1,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=4,
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
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm')
                )
            )
        ),
        
        # Attention Bias配置（新方案）
        attention_bias_config=dict(
            type='AttentionBiasGenerator',
            bev_feature_shape=(128, 128),
            pers_feature_shape=(6, 20, 50),
            window_size=8,
            bias_scale=2.5,
            learnable_scale=True,
            min_scale=0.5,
            max_scale=5.0,
            use_local_bias=True,
            use_gaussian_window=False,
            gaussian_sigma=2.0,
            debug_print=True,                # 🔥 启用调试打印
            print_interval=100,              # 🔥 每100个iteration打印一次
            fp16=True
        ),
        
        # 旧方案配置（兼容性保留）
        renderer_config=dict(
            render_method='gaussian',
            gaussian_sigma=1.0,
            bev_feature_shape=(128, 128),
            pers_feature_shape=(6, 20, 50),
            normalize_weights=True
        ),
        
        modulator_config=dict(
            type='FeatureModulator',
            modulation_type='element_wise',
            normalize_weights=False,
            residual_connection=True,
            residual_weight=0.7,
            learnable_modulation=False,
            activation='none'
        ),
    )
)

# 🔥 优化器配置：冻结骨干+AQR学习
optimizer = dict(
    type='AdamW',
    lr=0.00014,
    paramwise_cfg=dict(
        custom_keys={
            # 预训练骨干：完全冻结
            'img_backbone': dict(lr_mult=0.0),
            'pts_backbone': dict(lr_mult=0.0),
            'pts_voxel_encoder': dict(lr_mult=0.0),
            'pts_middle_encoder': dict(lr_mult=0.0),
            
            # Neck层：极低学习率微调
            'img_neck': dict(lr_mult=0.05),
            'pts_neck': dict(lr_mult=0.05),
            
            # CMT核心组件：适度学习
            'transformer': dict(lr_mult=0.5),
            'query_embed': dict(lr_mult=0.5),
            'reference_points': dict(lr_mult=0.3),
            'task_heads': dict(lr_mult=0.8),
            'shared_conv': dict(lr_mult=0.5),
            
            # AQR新增组件：正常学习
            'aqr_weight_generator': dict(lr_mult=1.0),
            'attention_bias_generator': dict(lr_mult=1.0),
            'attention_bias_generator.bias_scale': dict(lr_mult=0.5),
        }
    ),
    weight_decay=0.01
)

# DDP配置
find_unused_parameters = True

# 模型冻结配置
model = dict(
    img_backbone=dict(frozen_stages=4, norm_eval=True),  # ResNet50 fully frozen
    pts_backbone=dict(frozen_stages=3),                  # SECOND fully frozen
    img_neck=dict(norm_eval=True),
    pts_neck=dict(norm_eval=True),
)

