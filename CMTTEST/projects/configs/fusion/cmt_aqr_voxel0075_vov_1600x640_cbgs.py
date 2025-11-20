# 基于AQR Attention Bias机制的CMT配置文件 (1600×640分辨率)
# 继承voxel0075_vov_1600x640基础配置1

# 继承原始配置
_base_ = ['./cmt_voxel0075_vov_1600x640_cbgs.py']



# 🔥 优化器配置：冻结骨干+AQR学习
optimizer = dict(
    type='AdamW',
    lr=0.0002,  # 🔥 1600x640分辨率使用稍高的学习率（800x320用0.00014，这里用0.0002）
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
            'bias_scale': dict(lr_mult=2.0),                        # 🔥 bias_scale：200%学习率（加快学习）
            'weight_renderer': dict(lr_mult=1.0),                   # 权重渲染器：100%学习率（旧方案）
            'feature_modulator': dict(lr_mult=1.0),                 # 特征调制器：100%学习率（旧方案）
        }
    ),
    weight_decay=0.01
)




# 🔥 模型冻结配置
model = dict(
    # === 图像骨干冻结（VoVNet）===
    img_backbone=dict(
        frozen_stages=4,    # 🔥 VoVNet完全冻结（stage 1,2,3,4全部冻结）
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



# 🔥 核心修改：直接在CmtHead中启用AQR功能
model = dict(
    pts_bbox_head=dict(
        _delete_=False,  # ✅ 配置合并标记（不删除base配置，只覆盖指定字段）
        # 🔥 不重复定义type，从base继承 type='CmtHead'
        
        # === AQR特定配置 ===
        enable_aqr=True,               # ✅ 启用AQR机制（对比实验：AQR模型）
        debug_mode=True,              # 调试模式（生产环境建议关闭）
        visualization_interval=1000,    # 可视化间隔
        
        # AQR权重生成器配置
        aqr_config=dict(
            _delete_=True,  # 🔥 删除base中的None，使用完整的新配置
            embed_dims=256,           # 嵌入维度
            window_sizes=[15, 5],     # 🔥 [camera_window, lidar_window] - 针对40×100特征图优化（原800x320用8，1600x640翻倍用15）
            use_type_embed=True,      # 使用类型嵌入
            bev_feature_shape=(180, 180),  # 🔥 BEV特征图尺寸（voxel_size=0.075, grid=1440, 1440/8=180）
            pers_feature_shape=(6, 40, 100),  # 🔥 透视特征图尺寸（1600x640, 1600/16=100, 640/16=40）
            encoder_config=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=1,         # 权重生成只需1层
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='PETRMultiheadFlashAttention',  # 🔥 使用FlashAttention优化（标准版本）
                            embed_dims=256,
                            num_heads=4,      # 权重生成使用较少的注意力头
                            dropout=0.1,
                            # use_flashbias=True  # ❌ 不需要！AQR权重生成器没有attention_bias
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
        
        # 权重图渲染器配置（旧方案，已废弃）
        renderer_config=dict(
            _delete_=True,  # 🔥 删除base中的None，使用完整的新配置
            render_method='gaussian',      # 渲染方法: ['gaussian', 'bilinear', 'direct', 'distance_weighted']
            gaussian_sigma=2.0,            # 🔥 针对大特征图优化（800x320用1.0，1600x640用2.0）
            bilinear_radius=2.0,           # 双线性插值半径（增大）
            distance_decay=0.8,            # 距离衰减因子
            min_weight_threshold=0.01,     # 最小权重阈值
            bev_feature_shape=(180, 180),  # 🔥 BEV特征图尺寸（基于voxel_size=0.075）
            pers_feature_shape=(6, 40, 100), # 🔥 透视特征图尺寸 (views, h, w) - 针对1600×640
            normalize_weights=True         # 使用轻度裁剪
        ),
        
        # 🔥 特征调制模式选择（旧方案，已废弃，使用attention_bias替代）
        use_simple_modulation=False,     # False=完整模式(推荐), True=简化模式
        
        # 特征调制器配置（旧方案，仅在完整模式use_simple_modulation=False时使用）
        modulator_config=dict(
            _delete_=True,  # 🔥 删除base中的None，使用完整的新配置
            type='FeatureModulator',
            modulation_type='element_wise',  # 调制类型: ['element_wise', 'channel_wise', 'adaptive']
            normalize_weights=False,         # 禁用FeatureModulator内部的归一化（WeightRenderer已处理）
            residual_connection=True,        # 🛡️ 残差连接（防止特征消失）
            residual_weight=0.7,             # 🔥 强化残差保护：保留70%原始特征
            learnable_modulation=False,      # 可学习调制参数
            activation='none'                # 激活函数: ['none', 'sigmoid', 'tanh', 'relu']
        ),
        
        # 🔥 关键修复：使用支持attention_bias的PETRMultiheadAttention（而非FlashAttention）
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
                        # Self-attention（保持FlashAttention优化）
                        dict(
                            type='PETRMultiheadFlashAttention',  # 🔥 使用FlashAttention优化
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                            # use_flashbias=True  # 可选：Cross-attention需要时才启用
                        ),
                        # 🔥 Cross-attention：使用FlashAttention（支持attention_bias）
                        dict(
                            type='PETRMultiheadFlashAttention',  # 🔥 使用FlashAttention优化
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                            # use_flashbias=True  # 可选：需要支持attention_bias时才启用
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
        
        # 🔥 Attention Bias配置（新方案，推荐）
        attention_bias_config=dict(
            _delete_=True,  # 🔥 删除base中的None，使用完整的新配置
            type='AttentionBiasGenerator',
            bev_feature_shape=(180, 180),    # BEV特征图尺寸（与renderer_config保持一致）
            pers_feature_shape=(6, 40, 100),  # 透视特征图尺寸（与renderer_config保持一致）
            window_size=15,                  # 🔥 局部窗口大小（与camera_window=15保持一致，800x320用8，1600x640约翻倍）
            bias_scale=2.5,                  # 🔥 bias缩放因子初始值
            learnable_scale=True,            # 🔥 让bias_scale可学习（推荐）
            min_scale=0.5,                   # 🔥 最小scale（防止退化）
            max_scale=5.0,                   # 🔥 最大scale（防止softmax饱和）
            use_local_bias=True,             # 使用局部窗口bias（推荐）
            use_gaussian_window=False,       # 🔥 是否使用高斯衰减窗口（False=均匀窗口，True=高斯衰减）
            gaussian_sigma=2.5,              # 🔥 高斯核标准差（仅use_gaussian_window=True时生效，1600x640用2.5）
            debug_print=True,                # 🔥 启用调试打印（显示bias统计信息）
            print_interval=1000,             # 🔥 每1000个iteration打印一次
            fp16=True                        # 使用FP16以节省内存
        ),
    )
)



# 🔥 DDP配置：允许未使用参数（AQR组件可能在某些情况下不参与梯度计算）
find_unused_parameters = True

