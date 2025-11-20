# 🎯 局部Attention Bias实现设计方案

**创建时间**: 2025-01-XX  
**状态**: 实现设计阶段  
**目标**: 为CMT的Transformer cross-attention添加空间感知的局部bias机制

---

## 📐 窗口大小设计

### 1. **物理空间分析**

```
BEV特征图规格：
- 尺寸：180 × 180
- 覆盖范围：108m × 108m（-54m 到 +54m）
- 每像素代表：0.6m × 0.6m

不同类别目标的空间尺度：
┌─────────────┬──────────┬────────────┬──────────────┐
│  目标类别   │ 真实尺寸 │ 特征图尺寸 │ 推荐窗口    │
├─────────────┼──────────┼────────────┼──────────────┤
│ Car         │ 4×2m     │ 7×3像素    │ 11×11 可覆盖 │
│ Bus/Trailer │ 12×3m    │ 20×5像素   │ 25×25 可覆盖 │
│ Pedestrian  │ 1×0.5m   │ 2×1像素    │ 5×5 可覆盖   │
│ Barrier     │ 4×0.5m   │ 7×1像素    │ 11×11 可覆盖 │
│ Bicycle     │ 2×0.7m   │ 3×1像素    │ 7×7 可覆盖   │
└─────────────┴──────────┴────────────┴──────────────┘
```

### 2. **窗口大小策略对比**

```python
# 方案A：保守窗口（对标LAM的lidar window）⭐⭐⭐
BEV_WINDOW_SIZE = 5
Camera_WINDOW_SIZE = 5
覆盖范围：5 × 0.6m = 3.0m
优点：计算量小、精确聚焦
缺点：无法覆盖大目标（如Bus）

# 方案B：中等窗口（对标LAM的camera window）⭐⭐⭐⭐⭐ 【推荐】
BEV_WINDOW_SIZE = 15
Camera_WINDOW_SIZE = 15
覆盖范围：15 × 0.6m = 9.0m
优点：能覆盖绝大多数目标、与LAM一致、平衡性好
缺点：较大目标边缘可能不完整

# 方案C：大窗口（激进）⭐⭐⭐
BEV_WINDOW_SIZE = 25
Camera_WINDOW_SIZE = 25
覆盖范围：25 × 0.6m = 15.0m
优点：覆盖所有目标
缺点：接近全局、失去局部性优势

# 方案D：自适应窗口（理想但复杂）⭐⭐⭐⭐
根据query的预测尺寸动态调整窗口大小
优点：理论最优
缺点：实现复杂、难以向量化
```

**🎯 最终选择：方案B（window_size=15）**
- 与AQR的LAM camera window一致
- 能覆盖90%以上的目标
- 保持合理的计算开销

---

## 🏗️ 架构设计

### 整体流程图

```
┌──────────────────────────────────────────────────────────────┐
│                   CmtHead Forward                            │
└──────────────────────────────────────────────────────────────┘
                            │
                            ├──> 1. 特征提取（x, x_img）
                            │
                            ├──> 2. 参考点获取（reference_points）
                            │
                            ├──> 3. 计算投影位置（pts_bev, pts_pers_idx）
                            │
                            ├──> 4. 生成AQR权重
                            │      └─> AQRWeightGenerator
                            │           ├─> lidar_weights [bs, num_queries]
                            │           └─> camera_weights [bs, num_queries]
                            │
                            ├──> 5. ✨ 生成局部Attention Bias ✨
                            │      └─> AttentionBiasGenerator
                            │           ├─> 输入：weights, positions, window_size
                            │           ├─> 计算局部窗口mask
                            │           └─> 输出：attention_bias [bs, num_queries, num_features]
                            │
                            ├──> 6. 调用Transformer
                            │      └─> self.transformer(
                            │            x, x_img, query_embeds,
                            │            bev_pos_embeds, rv_pos_embeds,
                            │            attention_bias=bias  ← ✨ 新参数 ✨
                            │          )
                            │
                            └──> 7. 检测头输出
```

---

## 🔧 核心模块设计

### 1. **AttentionBiasGenerator 模块**

```python
class AttentionBiasGenerator(nn.Module):
    """
    局部注意力bias生成器
    
    功能：
    1. 根据query的空间位置生成局部窗口
    2. 将per-query权重扩散到局部窗口
    3. 生成 [bs, num_queries, num_features] 的bias矩阵
    """
    
    def __init__(self,
                 bev_feature_shape=(180, 180),
                 pers_feature_shape=(6, 40, 100),
                 window_size=15,
                 bias_scale=1.0):
        """
        Args:
            bev_feature_shape: BEV特征图尺寸
            pers_feature_shape: 透视特征图尺寸
            window_size: 局部窗口大小
            bias_scale: bias缩放因子
        """
        
    def forward(self, 
                lidar_weights,      # [bs, num_queries]
                camera_weights,     # [bs, num_queries]
                pts_bev_indices,    # [bs, num_queries] BEV位置索引
                pts_pers_indices):  # [bs, num_queries, 3] 透视位置索引(view,h,w)
        """
        生成局部attention bias
        
        Returns:
            attention_bias: [bs, num_queries, total_features]
                其中 total_features = bev_h*bev_w + 6*pers_h*pers_w
        """
```

### 2. **窗口生成逻辑**

```python
def _generate_local_window_bias(self, 
                                 weights,        # [bs, num_queries]
                                 positions,      # [bs, num_queries] 或 [bs, num_queries, 3]
                                 feature_shape,  # (H, W) 或 (V, H, W)
                                 window_size):   # int
    """
    核心算法：向量化的局部窗口bias生成
    
    步骤：
    1. 计算窗口偏移量 offsets
    2. 生成有效窗口索引 valid_indices
    3. 创建bias矩阵并填充
    4. 应用权重缩放
    """
    
    # 伪代码示例
    batch_size, num_queries = weights.shape
    H, W = feature_shape[-2:]
    total_features = H * W
    
    # 1. 生成窗口偏移（meshgrid）
    offsets = torch.arange(-window_size//2, window_size//2+1)
    y_off, x_off = torch.meshgrid(offsets, offsets)
    window_offsets = y_off * W + x_off  # [window_size^2]
    
    # 2. 计算所有query的窗口索引（向量化）
    query_indices = positions.unsqueeze(-1) + window_offsets.view(1, 1, -1)
    # query_indices: [bs, num_queries, window_size^2]
    
    # 3. 边界检查
    valid_mask = (query_indices >= 0) & (query_indices < total_features)
    
    # 4. 创建bias矩阵
    bias = torch.zeros(batch_size, num_queries, total_features, device=weights.device)
    
    # 5. 填充局部窗口（使用scatter_add）
    weights_expanded = weights.unsqueeze(-1).expand(-1, -1, window_size**2)
    weights_masked = torch.where(valid_mask, weights_expanded, torch.zeros_like(weights_expanded))
    
    # 6. 向量化填充
    bias.scatter_add_(
        dim=2,
        index=query_indices.clamp(0, total_features-1),
        src=weights_masked
    )
    
    return bias
```

---

## 💡 关键实现细节

### 1. **向量化优化**

```python
# ❌ 低效的循环实现
for b in range(batch_size):
    for q in range(num_queries):
        for offset in window_offsets:
            idx = positions[b, q] + offset
            if 0 <= idx < total_features:
                bias[b, q, idx] = weights[b, q]

# ✅ 高效的向量化实现
query_indices = positions.unsqueeze(-1) + window_offsets.view(1, 1, -1)
valid_mask = (query_indices >= 0) & (query_indices < total_features)
weights_expanded = weights.unsqueeze(-1).expand(-1, -1, window_size**2)
bias.scatter_add_(dim=2, index=query_indices.clamp(0), src=weights_masked)
```

### 2. **边界处理**

```python
# BEV边界检查（2D网格）
query_y = positions // W
query_x = positions % W

window_y = query_y.unsqueeze(-1) + y_offsets.view(1, 1, -1)
window_x = query_x.unsqueeze(-1) + x_offsets.view(1, 1, -1)

valid_y = (window_y >= 0) & (window_y < H)
valid_x = (window_x >= 0) & (window_x < W)
valid_mask = valid_y & valid_x

# Camera边界检查（考虑多视角）
view_mask = (view_indices >= 0) & (view_indices < 6)
valid_mask = valid_mask & view_mask
```

### 3. **内存优化**

```python
# 计算内存占用
batch_size = 2
num_queries = 900
total_features = 180*180 + 6*40*100 = 32400 + 24000 = 56400

bias_memory = batch_size * num_queries * total_features * 4 bytes (fp32)
            = 2 * 900 * 56400 * 4 / 1024^2
            ≈ 387 MB

# 优化策略：
1. 使用fp16存储：减半至 ~194 MB
2. 稀疏表示：只存储局部窗口
3. 按需计算：分batch计算
```

---

## 🔗 集成点设计

### 修改 CmtTransformer

```python
@TRANSFORMER.register_module()
class CmtTransformer(BaseModule):
    
    def forward(self, x, x_img, query_embed, 
                bev_pos_embed, rv_pos_embed, 
                attn_masks=None, 
                attention_bias=None,  # ← ✨ 新增参数
                reg_branch=None):
        """
        Args:
            attention_bias: [bs, num_queries, total_features] 或 None
        """
        
        # 融合memory
        memory = torch.cat([bev_memory, rv_memory], dim=0)
        pos_embed = torch.cat([bev_pos_embed, rv_pos_embed], dim=0)
        
        # 传递bias到decoder
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            attn_masks=[attn_masks, None],
            attention_bias=attention_bias,  # ← ✨ 传递bias
            reg_branch=reg_branch
        )
        
        return out_dec, memory
```

### 修改 PETRTransformerDecoder

```python
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerDecoder(TransformerLayerSequence):
    
    def forward(self, query, *args, 
                attention_bias=None,  # ← ✨ 新增
                **kwargs):
        """
        Args:
            attention_bias: [bs, num_queries, num_features]
        """
        
        # 调整维度：[bs, num_queries, num_features] → [num_queries, bs, num_features]
        if attention_bias is not None:
            attention_bias = attention_bias.transpose(0, 1)
        
        # 逐层传递
        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, 
                         attention_bias=attention_bias,  # ← ✨ 传递
                         **kwargs)
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))
        
        return torch.stack(intermediate)
```

### 修改 PETRTransformerDecoderLayer

```python
@TRANSFORMER_LAYER.register_module()
class PETRTransformerDecoderLayer(BaseTransformerLayer):
    
    def forward(self, query, key=None, value=None,
                query_pos=None, key_pos=None,
                attn_masks=None, 
                attention_bias=None,  # ← ✨ 新增
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """
        Args:
            attention_bias: [num_queries, bs, num_features]
        """
        
        # 在cross_attn时应用bias
        for layer in self.operation_order:
            if layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    attention_bias=attention_bias,  # ← ✨ 应用bias
                    key_padding_mask=key_padding_mask,
                    **kwargs
                )
                attn_index += 1
        
        return query
```

### 修改 MultiheadAttention

```python
class MultiheadAttention(nn.Module):
    
    def forward(self, query, key=None, value=None,
                query_pos=None, key_pos=None,
                attn_mask=None,
                attention_bias=None,  # ← ✨ 新增
                key_padding_mask=None,
                **kwargs):
        """
        Args:
            attention_bias: [num_queries, bs, num_features]
                在计算attention scores后加到scores上
        """
        
        # 标准attention计算
        q = k = self.qkv_proj(query)
        k = self.qkv_proj(key) if key is not None else k
        v = self.qkv_proj(value) if value is not None else k
        
        if query_pos is not None:
            q = q + self.qkv_proj(query_pos)
        if key_pos is not None:
            k = k + self.qkv_proj(key_pos)
        
        # 计算attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(d_k)
        # attn_scores: [bs, num_heads, num_queries, num_features]
        
        # ✨ 应用attention bias ✨
        if attention_bias is not None:
            # attention_bias: [num_queries, bs, num_features]
            # 需要调整维度并扩展到多头
            bias = attention_bias.transpose(0, 1)  # [bs, num_queries, num_features]
            bias = bias.unsqueeze(1)  # [bs, 1, num_queries, num_features]
            bias = bias.expand(-1, self.num_heads, -1, -1)  # [bs, num_heads, num_queries, num_features]
            
            attn_scores = attn_scores + bias  # ← ✨ 加bias
        
        # 应用mask
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask, float('-inf'))
        
        # Softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        output = torch.matmul(attn_weights, v)
        
        return output
```

---

## 📊 性能预估

### 计算复杂度

```
原始cross-attention：
- 时间复杂度：O(num_queries * num_features)
- 内存占用：O(bs * num_heads * num_queries * num_features)

添加局部bias：
- 额外时间：O(bs * num_queries * window_size^2)  ← 窗口生成
- 额外内存：O(bs * num_queries * num_features)   ← bias矩阵

window_size=15 时：
- 窗口生成：2 * 900 * 15^2 = 405,000 次操作（可向量化）
- bias矩阵：387 MB（fp32）或 194 MB（fp16）

结论：计算开销可接受，内存需要优化
```

### 优化建议

```python
# 1. 半精度存储
attention_bias = attention_bias.half()

# 2. 梯度检查点（如果内存紧张）
with torch.cuda.amp.autocast():
    bias = self.bias_generator(...)

# 3. 分块计算（如果batch_size很大）
for i in range(0, batch_size, chunk_size):
    bias_chunk = self.bias_generator(
        weights[i:i+chunk_size],
        positions[i:i+chunk_size]
    )
```

---

## 🎯 配置接口设计

```python
# 在配置文件中
model = dict(
    pts_head=dict(
        type='CmtHead',
        enable_aqr=True,
        
        # AQR权重生成配置
        aqr_config=dict(
            type='AQRWeightGenerator',
            embed_dims=256,
            window_sizes=[15, 5],  # LAM窗口
            ...
        ),
        
        # ✨ Attention Bias配置 ✨
        attention_bias_config=dict(
            enable=True,                    # 是否启用
            window_size=15,                 # 局部窗口大小
            bias_scale=1.0,                 # bias缩放因子
            use_local_bias=True,            # True=局部, False=全局
            fp16=True,                      # 是否使用fp16
            debug_mode=False                # 调试模式
        ),
        
        # 不再需要单独的renderer和modulator
        # renderer_config=None,  # 废弃
        # modulator_config=None, # 废弃
    )
)
```

---

## ✅ 实现检查清单

### Phase 1: 核心模块实现
- [ ] `AttentionBiasGenerator` 类实现
  - [ ] `__init__` 初始化
  - [ ] `_generate_bev_local_bias` BEV窗口bias
  - [ ] `_generate_camera_local_bias` Camera窗口bias
  - [ ] `forward` 主函数
  - [ ] 边界检查逻辑
  - [ ] 向量化优化

### Phase 2: Transformer集成
- [ ] 修改 `CmtTransformer.forward`
- [ ] 修改 `PETRTransformerDecoder.forward`
- [ ] 修改 `PETRTransformerDecoderLayer.forward`
- [ ] 修改 `MultiheadAttention.forward`

### Phase 3: CmtHead集成
- [ ] 添加 `attention_bias_config` 参数
- [ ] 初始化 `AttentionBiasGenerator`
- [ ] 在 `forward_single` 中生成bias
- [ ] 传递bias到Transformer

### Phase 4: 测试与优化
- [ ] 单元测试：窗口生成正确性
- [ ] 集成测试：端到端前向传播
- [ ] 性能测试：内存和速度
- [ ] 可视化：bias分布图

---

## 🚨 潜在风险

1. **内存占用**
   - 风险：387MB额外内存可能导致OOM
   - 缓解：使用fp16、分块计算

2. **数值稳定性**
   - 风险：bias过大导致softmax饱和
   - 缓解：添加`bias_scale`参数控制幅度

3. **Flash Attention兼容性**
   - 风险：Flash Attention可能不支持custom bias
   - 缓解：退回标准attention或修改bias应用方式

4. **训练不稳定**
   - 风险：初期bias可能扰乱已有的attention pattern
   - 缓解：渐进式启用（warmup）、较小的`bias_scale`

---

## 📖 参考文献

1. **Deformable DETR**: Conditional spatial attention
2. **DN-DETR**: Noise-based query denoising with positional bias
3. **Relative Position Bias**: Swin Transformer的相对位置编码
4. **Local Attention**: Longformer的滑动窗口机制

---

**主人，这个设计方案准备好了！接下来我们开始实现核心代码！** 🚀✨

