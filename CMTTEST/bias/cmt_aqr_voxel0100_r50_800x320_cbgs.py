# 基于AQR权重图渲染机制的CMT配置文件 (800×320分辨率)
# 继承voxel0100_r50_800x320基础配置

# 继承原始配置
_base_ = ['./cmt_voxel0100_r50_800x320_cbgs.py']

# 🔥 核心修改：直接在CmtHead中启用AQR功能
model = dict(
    pts_bbox_head=dict(
        type='CmtHead',     # 使用原始CmtHead，通过enable_aqr启用AQR功能
        
        # === AQR特定配置 ===
        enable_aqr=True,               # ✅ 启用AQR机制（对比实验：AQR模型）
        debug_mode=False,              # 调试模式（生产环境建议关闭）
        visualization_interval=100,    # 可视化间隔
        
        # AQR权重生成器配置
        aqr_config=dict(
            embed_dims=256,           # 嵌入维度
            window_sizes=[8, 5],      # 🔥 [camera_window, lidar_window] - 针对20×50特征图优化
            use_type_embed=True,      # 使用类型嵌入
            bev_feature_shape=(128, 128),  # 🔥 BEV特征图尺寸（voxel_size=0.1, grid=1024, 1024/8=128）
            pers_feature_shape=(6, 20, 50),  # 🔥 透视特征图尺寸（800x320, 800/16=50, 320/16=20）
            encoder_config=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=1,         # 权重生成只需1层
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=4,      # 权重生成使用较少的注意力头
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
        
        # 权重图渲染器配置
        renderer_config=dict(
            render_method='gaussian',      # 渲染方法: ['gaussian', 'bilinear', 'direct', 'distance_weighted']
            gaussian_sigma=1.0,            # 🔥 针对小特征图优化
            bilinear_radius=1.5,           # 双线性插值半径
            distance_decay=0.8,            # 距离衰减因子
            min_weight_threshold=0.01,     # 最小权重阈值
            bev_feature_shape=(128, 128),  # 🔥 BEV特征图尺寸（基于voxel_size=0.1）
            pers_feature_shape=(6, 20, 50), # 🔥 透视特征图尺寸 (views, h, w) - 针对800×320
            normalize_weights=True         # 使用轻度裁剪
        ),
        
        # 🔥 特征调制模式选择（旧方案，已废弃，使用attention_bias替代）
        use_simple_modulation=False,     # False=完整模式(推荐), True=简化模式
        
        # 特征调制器配置（旧方案，仅在完整模式use_simple_modulation=False时使用）
        modulator_config=dict(
            type='FeatureModulator',
            modulation_type='element_wise',  # 调制类型: ['element_wise', 'channel_wise', 'adaptive']
            normalize_weights=False,         # 禁用FeatureModulator内部的归一化（WeightRenderer已处理）
            residual_connection=True,        # 🛡️ 残差连接（防止特征消失）
            residual_weight=0.7,             # 🔥 强化残差保护：保留70%原始特征
            learnable_modulation=False,      # 可学习调制参数
            activation='none'                # 激活函数: ['none', 'sigmoid', 'tanh', 'relu']
        ),
        
        # 🔥 Attention Bias配置（新方案，推荐）
        attention_bias_config=dict(
            type='AttentionBiasGenerator',
            bev_feature_shape=(128, 128),    # BEV特征图尺寸（与renderer_config保持一致）
            pers_feature_shape=(6, 20, 50),  # 透视特征图尺寸（与renderer_config保持一致）
            window_size=8,                   # 局部窗口大小（与camera_window=8保持一致）
            bias_scale=2.5,                  # 🔥 bias缩放因子初始值
            learnable_scale=True,            # 🔥 让bias_scale可学习（推荐）
            min_scale=0.5,                   # 🔥 最小scale（防止退化）
            max_scale=5.0,                   # 🔥 最大scale（防止softmax饱和）
            use_local_bias=True,             # 使用局部窗口bias（推荐）
            fp16=True                        # 使用FP16以节省内存
        ),
    )
)

# 🔥 优化器配置：冻结骨干+AQR学习
optimizer = dict(
    type='AdamW',
    lr=0.00014,  # 保持原始学习率
    paramwise_cfg=dict(
        custom_keys={
            # === 预训练骨干：完全冻结（使用CMT预训练权重）===
            'img_backbone': dict(lr_mult=0.0),          # 🔥 图像骨干：完全冻结
            'pts_backbone': dict(lr_mult=0.0),          # 🔥 点云骨干：完全冻结
            'pts_voxel_encoder': dict(lr_mult=0.0),     # 🔥 点云编码器：完全冻结
            'pts_middle_encoder': dict(lr_mult=0.0),    # 🔥 中间编码器：完全冻结
            
            # === Neck层：极低学习率微调（适应AQR调制）===
            'img_neck': dict(lr_mult=0.05),             # 图像颈部：5%学习率
            'pts_neck': dict(lr_mult=0.05),             # 点云颈部：5%学习率
            
            # === CMT核心组件：适度学习（适应Attention Bias）===
            'transformer': dict(lr_mult=0.5),           # 🔥 Transformer：50%学习率（需要适应bias）
            'query_embed': dict(lr_mult=0.5),           # 🔥 查询嵌入：50%学习率
            'reference_points': dict(lr_mult=0.3),      # 🔥 参考点：30%学习率（更保守）
            'task_heads': dict(lr_mult=0.8),            # 🔥 任务头：80%学习率
            'shared_conv': dict(lr_mult=0.5),           # 🔥 共享卷积：50%学习率
            
            # === AQR新增组件：正常学习 ===
            'aqr_weight_generator': dict(lr_mult=1.0),              # AQR权重生成器：100%学习率
            'attention_bias_generator': dict(lr_mult=1.0),          # 🔥 Attention Bias生成器：100%学习率
            'attention_bias_generator.bias_scale': dict(lr_mult=0.5), # 🔥 bias_scale：50%学习率（更稳定）
            'weight_renderer': dict(lr_mult=1.0),                   # 权重渲染器：100%学习率（旧方案）
            'feature_modulator': dict(lr_mult=1.0),                 # 特征调制器：100%学习率（旧方案）
        }
    ),
    weight_decay=0.01
)

# 🔥 DDP配置：允许未使用参数（AQR组件可能在某些情况下不参与梯度计算）
find_unused_parameters = True

# 🔥 模型冻结配置
model = dict(
    # === 图像骨干冻结（ResNet50）===
    img_backbone=dict(
        frozen_stages=4,    # 🔥 ResNet50完全冻结（stage 0,1,2,3全部冻结）
        norm_eval=True,     # 🔥 BN层保持eval模式（不更新统计量）
    ),
    
    # === 点云骨干冻结（SECOND）===
    pts_backbone=dict(
        frozen_stages=3,    # 🔥 SECOND完全冻结（3层全部冻结）
    ),
    
    # === Neck层（选择性微调）===
    img_neck=dict(
        norm_eval=True,     # BN保持eval模式
    ),
    pts_neck=dict(
        norm_eval=True,     # BN保持eval模式
    ),
)

