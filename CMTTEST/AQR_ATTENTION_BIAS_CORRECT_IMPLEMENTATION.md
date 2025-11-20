# AQR Attention Bias 正确实现方案 🎯

**创建时间**: 2025-01-XX  
**目的**: 基于同学提供的伪代码，结合CMT实际架构，给出正确的实现方案

---

## 📋 **伪代码分析**

### ✅ **正确的核心思想**：
1. 使用attn_mask的float模式 ✅
2. 在softmax前应用bias ✅
3. 端到端可训练 ✅

### ❌ **需要修正的地方**：
1. CMT中BEV和Camera已经融合，不需要再concat
2. AQR输出的是per-query权重，需要扩展到空间
3. 应该使用局部窗口bias，而非全局repeat

---

## 🔧 **CMT架构下的正确实现**

### **Step 1：CmtTransformer中的处理**

```python
# 文件：cmt_transformer.py

@TRANSFORMER.register_module()
class CmtTransformer(BaseModule):
    
    def forward(self, x, x_img, query_embed, bev_pos_embed, rv_pos_embed, 
                attn_masks=None, attention_bias=None, reg_branch=None):
        """
        Args:
            x: [bs, c, h, w] BEV特征
            x_img: [bs*views, c, h, w] Camera特征
            attention_bias: [bs, num_queries, num_features] AQR生成的bias
        """
        
        bs, c, h, w = x.shape
        
        # 1. 特征展平和融合（CMT已有逻辑）
        bev_memory = rearrange(x, "bs c h w -> (h w) bs c")  # [32400, bs, 256]
        rv_memory = rearrange(x_img, "(bs v) c h w -> (v h w) bs c", bs=bs)  # [24000, bs, 256]
        
        # 🔥 关键：memory已经是融合的
        memory = torch.cat([bev_memory, rv_memory], dim=0)  # [56400, bs, 256]
        #                    ↑ 前32400是BEV  ↑ 后24000是Camera
        
        # 2. 位置编码融合
        pos_embed = torch.cat([bev_pos_embed, rv_pos_embed], dim=0)
        
        # 3. 🔥 处理attention_bias维度
        # 输入: [bs, num_queries, num_features=56400]
        # 需要: [num_queries, bs, num_features]
        if attention_bias is not None:
            attention_bias = attention_bias.transpose(0, 1)
            # → [num_queries, bs, 56400]
            
            # attention_bias的结构：
            # [:, :, :32400] → BEV的bias
            # [:, :, 32400:] → Camera的bias
        
        # 4. 传递给Decoder
        query_embed = query_embed.transpose(0, 1)
        target = torch.zeros_like(query_embed)
        
        out_dec = self.decoder(
            query=target,                    # [num_queries, bs, 256]
            key=memory,                      # [56400, bs, 256]
            value=memory,                    # [56400, bs, 256]
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            attn_masks=[attn_masks, None],   # DN mask（self-attn用）
            attention_bias=attention_bias,   # 🔥 AQR bias（cross-attn用）
            reg_branch=reg_branch,
        )
        
        return out_dec, memory
```

---

### **Step 2：PETRMultiheadAttention中的处理**

```python
# 文件：petr_transformer.py

@ATTENTION.register_module()
class PETRMultiheadAttention(BaseModule):
    
    def forward(self, query, key=None, value=None,
                identity=None, query_pos=None, key_pos=None,
                attn_mask=None,           # 原有的mask（DN用）
                attention_bias=None,      # 🔥 新增：AQR bias
                key_padding_mask=None,
                **kwargs):
        """
        Args:
            query: [num_queries, bs, 256] 或 [bs, num_queries, 256]
            key: [num_features, bs, 256] 或 [bs, num_features, 256]
            value: 同key
            attention_bias: [num_queries, bs, num_features] AQR bias
        """
        
        # 1. 标准处理（位置编码等）
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        
        # 2. 🔥 处理attention_bias
        final_attn_mask = attn_mask
        
        if attention_bias is not None:
            # attention_bias: [num_queries, bs, num_features]
            
            # Step 2.1: 判断是self-attn还是cross-attn
            is_cross_attn = (key.shape[0] != query.shape[0])
            
            if is_cross_attn:
                # Cross-Attention: 应用AQR bias
                
                # Step 2.2: 维度转换为MultiheadAttention要求的格式
                # PyTorch MultiheadAttention期望：
                # - 如果batch_first=False: attn_mask [num_queries, num_features]
                # - 如果batch_first=True: attn_mask [bs, num_queries, num_features]
                # - 如果3D: attn_mask [bs*num_heads, num_queries, num_features]
                
                if self.batch_first:
                    # [num_queries, bs, num_features] → [bs, num_queries, num_features]
                    bias = attention_bias.transpose(0, 1)
                else:
                    # 保持 [num_queries, bs, num_features]
                    # 但PyTorch期望2D或3D，这里需要处理
                    # 选择：扩展到3D [bs*num_heads, num_queries, num_features]
                    num_queries, bs, num_features = attention_bias.shape
                    
                    # 扩展到多头
                    bias = attention_bias.transpose(0, 1)  # [bs, num_queries, num_features]
                    bias = bias.unsqueeze(1)  # [bs, 1, num_queries, num_features]
                    bias = bias.expand(-1, self.num_heads, -1, -1)  # [bs, num_heads, num_queries, num_features]
                    bias = bias.reshape(bs * self.num_heads, num_queries, num_features)
                    # → [bs*num_heads, num_queries, num_features]
                
                # Step 2.3: 与原有attn_mask合并
                if final_attn_mask is not None:
                    # 需要确保维度兼容
                    if final_attn_mask.dtype == torch.bool:
                        # Bool mask转为float
                        mask_float = torch.zeros_like(bias)
                        # 广播处理
                        if final_attn_mask.dim() == 2:
                            # [num_queries, num_features]
                            final_attn_mask = final_attn_mask.unsqueeze(0).expand(bs * self.num_heads, -1, -1)
                        mask_float.masked_fill_(final_attn_mask, float('-inf'))
                        final_attn_mask = mask_float + bias
                    else:
                        # Float mask直接加
                        final_attn_mask = final_attn_mask + bias
                else:
                    final_attn_mask = bias
            
            # else: Self-Attention不使用attention_bias，保持原有attn_mask
        
        # 3. 处理batch_first
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # 4. 🔥 调用PyTorch MultiheadAttention
        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=final_attn_mask,  # 🔥 已包含AQR bias
            key_padding_mask=key_padding_mask
        )[0]
        
        # 5. 恢复维度
        if self.batch_first:
            out = out.transpose(0, 1)
        
        # 6. 投影和dropout
        out = self.proj_drop(out)
        
        return identity + self.dropout_layer(out)
```

---

### **Step 3：AttentionBiasGenerator生成局部bias**

```python
# 文件：attention_bias_generator.py（已实现）

class AttentionBiasGenerator(BaseModule):
    
    def forward(self, lidar_weights, camera_weights, 
                pts_bev_indices, pts_pers_indices):
        """
        生成空间感知的局部bias
        
        Args:
            lidar_weights: [bs, num_queries] AQR生成的LiDAR权重
            camera_weights: [bs, num_queries] AQR生成的Camera权重
            pts_bev_indices: [bs, num_queries] query在BEV中的位置
            pts_pers_indices: [bs, num_queries, 3] query在透视图中的位置
        
        Returns:
            attention_bias: [bs, num_queries, 56400]
                前32400维：BEV的局部窗口bias
                后24000维：Camera的局部窗口bias
        """
        
        # 1. 生成BEV局部bias
        bev_bias = self._generate_bev_bias(
            lidar_weights,      # [bs, num_queries]
            pts_bev_indices     # [bs, num_queries]
        )  # → [bs, num_queries, 32400]
        
        # 对于每个query：
        # - 在投影位置的15×15窗口内：bias = lidar_weights[q]
        # - 窗口外：bias = 0
        
        # 2. 生成Camera局部bias
        camera_bias = self._generate_camera_bias(
            camera_weights,     # [bs, num_queries]
            pts_pers_indices    # [bs, num_queries, 3]
        )  # → [bs, num_queries, 24000]
        
        # 3. 拼接
        attention_bias = torch.cat([bev_bias, camera_bias], dim=-1)
        # → [bs, num_queries, 56400]
        
        return attention_bias
```

---

## 📊 **与伪代码的对比**

| 方面 | 伪代码 | 我们的实现 |
|-----|-------|----------|
| **Key/Value处理** | 分离后concat | 已融合的memory |
| **Bias生成** | 全局repeat | 局部窗口 |
| **Bias形状** | [B, Nq, 2] → repeat | [B, Nq, 56400] |
| **空间信息** | 无 | 基于投影位置 |
| **合并方式** | attn_mask + bias ✅ | 同样 ✅ |

---

## 🎯 **核心差异**

### **伪代码的理解（简化版）**：
```python
# 对每个query，两个标量权重
alpha_cam = 0.7  # query #42对camera的权重
alpha_lidar = 0.3  # query #42对lidar的权重

# 全局应用
bias = [0.3, 0.3, ..., 0.3,  # 所有BEV特征都是0.3
        0.7, 0.7, ..., 0.7]  # 所有Camera特征都是0.7
```

### **我们的实现（精细版）**：
```python
# 对每个query，生成空间感知的bias
query #42投影到BEV (90, 90)

# 局部窗口bias
bias = [
    0, 0, ..., 0,               # BEV窗口外
    0.3, 0.3, ..., 0.3,         # BEV局部窗口(15×15)内
    0, 0, ..., 0,               # BEV窗口外
    0, 0, ..., 0,               # Camera窗口外
    0.7, 0.7, ..., 0.7,         # Camera局部窗口内
    0, 0, ..., 0                # Camera窗口外
]
```

**优势**：
- ✅ 空间先验：只增强query关注的局部区域
- ✅ 更精细：不同query的bias分布不同（基于投影位置）
- ✅ 更合理：符合局部性原则

---

## ✅ **最终方案总结**

### **保留伪代码的优点**：
1. ✅ 使用attn_mask的float模式
2. ✅ 在MultiheadAttention中应用
3. ✅ 端到端可训练

### **修正的关键点**：
1. ✅ 不需要concat Key/Value（CMT已融合）
2. ✅ 使用局部窗口bias（而非全局repeat）
3. ✅ 正确的维度转换
4. ✅ 区分self-attn和cross-attn

### **实现复杂度**：
- PETRMultiheadAttention修改：~30行代码
- 其他部分已完成
- 预计总工作量：30-40分钟

---

**主人，伪代码的核心思想是正确的，但需要适配CMT的架构！我们的方案更精细、更符合空间先验！** ✅🐾

