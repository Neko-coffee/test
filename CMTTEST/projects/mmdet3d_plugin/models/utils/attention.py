# Copyright (c) 2023 megvii-model. All Rights Reserved.

import math
import torch
import torch.nn as nn
from torch.nn.init import (
    xavier_uniform_,
    constant_,
    xavier_normal_
)
from torch.nn.functional import linear
import sys
import os

from einops import rearrange
from mmcv.runner import auto_fp16
from mmcv.runner.base_module import BaseModule

# 🔥 FlashBias 导入
FLASHBIAS_AVAILABLE = False
FLASHBIAS_VERSION = None

try:
    # 尝试从 external/FlashBias 导入
    flashbias_path = os.path.join(os.path.dirname(__file__), '../../../../external/FlashBias')
    flashbias_abs_path = os.path.abspath(flashbias_path)
    
    print(f"🔍 尝试从路径导入 FlashBias: {flashbias_abs_path}")
    
    if os.path.exists(flashbias_abs_path):
        print(f"✅ FlashBias 路径存在: {flashbias_abs_path}")
        sys.path.insert(0, flashbias_abs_path)
        
        # 检查文件是否存在
        flash_bias_file = os.path.join(flashbias_abs_path, 'flash_bias_triton.py')
        if os.path.exists(flash_bias_file):
            print(f"✅ FlashBias 文件存在: {flash_bias_file}")
            from flash_bias_triton import flash_bias_func
            FLASHBIAS_AVAILABLE = True
            FLASHBIAS_VERSION = "triton"
            print("✅ FlashBias (Triton) loaded successfully!")
        else:
            print(f"❌ FlashBias 文件不存在: {flash_bias_file}")
            raise ImportError("flash_bias_triton.py not found")
    else:
        print(f"❌ FlashBias 路径不存在: {flashbias_abs_path}")
        # 尝试从系统路径导入
        try:
            from flash_bias_triton import flash_bias_func
            FLASHBIAS_AVAILABLE = True
            FLASHBIAS_VERSION = "system"
            print("✅ FlashBias (System) loaded successfully!")
        except ImportError:
            print("⚠️ FlashBias not available in system path")
            raise ImportError("FlashBias not found in system path")
            
except ImportError as e:
    print(f"⚠️ FlashBias import failed: {e}")
    FLASHBIAS_AVAILABLE = False
    # 提供占位函数
    def flash_bias_func(*args, **kwargs):
        raise NotImplementedError("FlashBias not installed")


def _in_projection_packed(q, k, v, w, b=None):
    """输入投影的打包版本"""
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


class FlashBiasAttention(nn.Module):
    """
    FlashBias 注意力实现
    专门为 FlashBias 优化的注意力机制，支持 attention_bias
    
    特性：
    - 支持 Triton-based FlashBias（最佳性能）
    - 支持 PyTorch-SDPA-based FlashBias（兼容性）
    - 自动回退到标准注意力（保底）
    - 智能偏置转换（attn_bias → q_bias + k_bias）
    """
    
    def __init__(self, embed_dims, num_heads, dropout=0.0, bias=True, **kwargs):
        super(FlashBiasAttention, self).__init__()
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.dropout = dropout
        
        assert self.head_dim * num_heads == embed_dims, "embed_dims must be divisible by num_heads"
        
        # 输入投影
        self.in_proj_weight = nn.Parameter(torch.randn(3 * embed_dims, embed_dims))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.randn(3 * embed_dims))
        else:
            self.register_parameter('in_proj_bias', None)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dims, embed_dims, bias=bias)
        
        # FlashBias 可用性检查
        if not FLASHBIAS_AVAILABLE:
            print("⚠️ FlashBias not available, will use standard attention")

    # def forward(self, query, key, value, attn_mask=None, key_padding_mask=None,
    #             attn_bias=None, **kwargs):
    def forward(self, query=None, key=None, value=None, q=None, k=None, v=None, 
                attn_mask=None, key_padding_mask=None, attn_bias=None, **kwargs):
        """
        前向传播
        
        Args:
            query/q: [batch_size, seq_len, embed_dims] 查询
            key/k: [batch_size, seq_len, embed_dims] 键
            value/v: [batch_size, seq_len, embed_dims] 值
            attn_mask: 注意力掩码
            key_padding_mask: 键填充掩码
            attn_bias: [batch_size, num_heads, seq_len, seq_len] 注意力偏置
            
        Returns:
            context: [batch_size, seq_len, embed_dims] 输出
            attn_weights: 注意力权重（FlashBias 不返回）
        """
        # 🔥 参数兼容性处理：支持 q/k/v 和 query/key/value 两种调用方式
        if query is None and q is not None:
            query = q
        if key is None and k is not None:
            key = k
        if value is None and v is not None:
            value = v
            
        # 验证必要参数
        if query is None or key is None or value is None:
            raise ValueError("query/key/value 或 q/k/v 参数必须提供")

        batch_size, seq_len_q, embed_dims = query.shape
        _, seq_len_k, _ = key.shape
        _, seq_len_v, _ = value.shape
        
        # 🔥 保存原始 dtype（用于最后恢复）
        original_query_dtype = query.dtype

        # 🔥 调试信息
        # print(f"🔍 FlashBiasAttention Debug:")
        # print(f"   Input shapes: query={query.shape}, key={key.shape}, value={value.shape}")
        # print(f"   embed_dims={embed_dims}, num_heads={self.num_heads}, head_dim={self.head_dim}")
        # print(f"   seq_len_q={seq_len_q}, seq_len_k={seq_len_k}, seq_len_v={seq_len_v}")
        
        
        # 输入投影
        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        # 🔥 投影后调试信息
        # print(f"   After projection: q={q.shape}, k={k.shape}, v={v.shape}")
        
        # 验证维度匹配
        if embed_dims != self.num_heads * self.head_dim:
            print(f"⚠️ 维度不匹配: embed_dims={embed_dims} != num_heads*head_dim={self.num_heads * self.head_dim}")
            # 尝试自动调整
            if embed_dims % self.num_heads == 0:
                self.head_dim = embed_dims // self.num_heads
                print(f"🔧 自动调整 head_dim 为: {self.head_dim}")
            else:
                raise ValueError(f"embed_dims ({embed_dims}) 必须能被 num_heads ({self.num_heads}) 整除")
        
        
        # 重塑为多头格式 - 使用各自的序列长度
        try:
            q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)
            # print(f"   After reshape: q={q.shape}, k={k.shape}, v={v.shape}")
        except RuntimeError as e:
            print(f"❌ 重塑失败: {e}")
            print(f"   q size: {q.numel()}, expected: {batch_size * seq_len_q * self.num_heads * self.head_dim}")
            print(f"   k size: {k.numel()}, expected: {batch_size * seq_len_k * self.num_heads * self.head_dim}")
            print(f"   v size: {v.numel()}, expected: {batch_size * seq_len_v * self.num_heads * self.head_dim}")
            raise
        
        # 🔥 对于 FlashBias，确保输入是 fp16/bf16
        # 使用 attn_bias 的 dtype 作为目标（因为它来自 AQR，通常是 fp16）
        if attn_bias is not None and FLASHBIAS_AVAILABLE:
            # 使用 attn_bias 的 dtype 作为目标
            target_dtype = attn_bias.dtype
            
            # 如果 attn_bias 是 fp16/bf16，将 q/k/v 也转换为相同类型
            if target_dtype in [torch.float16, torch.bfloat16]:
                if q.dtype != target_dtype:
                    old_dtype = q.dtype
                    q = q.to(target_dtype)
                    k = k.to(target_dtype)
                    v = v.to(target_dtype)
                    print(f"🔧 已将 q/k/v 从 {old_dtype} 转换为 {target_dtype} 以匹配 FlashBias 要求")
            # 如果 attn_bias 不是 fp16/bf16，尝试转换为 fp16
            elif q.dtype in [torch.float16, torch.bfloat16]:
                # q 已经是 fp16/bf16，将 attn_bias 转换为匹配
                target_dtype = q.dtype
                attn_bias = attn_bias.to(target_dtype)
                print(f"🔧 已将 attn_bias 转换为 {target_dtype} 以匹配 q/k/v")
            else:
                # 都不是 fp16/bf16，默认转换为 fp16
                target_dtype = torch.float16
                q = q.half()
                k = k.half()
                v = v.half()
                attn_bias = attn_bias.half()
                print(f"🔧 已将 q/k/v/attn_bias 都转换为 fp16 以匹配 FlashBias 要求")
        
        # 🔥 注意力计算策略：AQR + FlashBias 是首要目标
        # 优先级：
        # 1. FlashBias (Triton) - 最优性能 + 支持 attention_bias
        # 2. PyTorch SDPA FlashBias - 备选方案
        # 3. PyTorch SDPA 标准 - 最终回退
        
        # 🔥 配置选项：是否使用 SVD + FlashBias（Triton）
        # 设置为 False 可以跳过 SVD，直接使用标准 SDPA（更快，显存占用稍高）
        USE_SVD_FLASHBIAS = False  # ← 改为 True 启用 SVD + FlashBias
        
        if FLASHBIAS_AVAILABLE and attn_bias is not None and USE_SVD_FLASHBIAS:
            # 🎯 方案1：AQR + FlashBias (Triton) - 需要 SVD
            try:
                print(f"🚀 开始调用 FlashBias (Triton)，q.shape={q.shape}, attn_bias.shape={attn_bias.shape}")
                context = self._flashbias_attention(q, k, v, attn_bias)
                print("✅ 使用 FlashBias (Triton) + AQR bias")
            except Exception as e:
                print(f"⚠️ FlashBias (Triton) 失败: {e}, 尝试 PyTorch-SDPA FlashBias")
                # 备选方案：PyTorch SDPA + concat 方式
                try:
                    context = self._pytorch_sdpa_attention(q, k, v, attn_bias)
                    print("✅ 使用 PyTorch-SDPA FlashBias + AQR bias")
                except Exception as e2:
                    print(f"⚠️ PyTorch-SDPA FlashBias 失败: {e2}, 回退到标准 SDPA")
                    # 最终回退：标准 SDPA + attn_mask
                    context = self._standard_attention(q, k, v, attn_bias)
                    print("✅ 使用标准 SDPA + AQR bias (回退)")
        elif attn_bias is not None:
            # 🎯 方案2：直接使用标准 SDPA（快速，无需 SVD）
            # PyTorch 2.1+ 的 SDPA 已经自动使用 FlashAttention
            context = self._standard_attention(q, k, v, attn_bias)
            # print("✅ 使用标准 SDPA + AQR bias (无 SVD)")
        elif FLASHBIAS_AVAILABLE:
            # 没有 bias 时，也使用 FlashBias 优化性能
            try:
                context = self._flashbias_attention(q, k, v, None)
            except Exception:
                context = self._standard_attention(q, k, v, None)
        else:
            # FlashBias 不可用，使用标准注意力
            context = self._standard_attention(q, k, v, attn_bias)
        
        # 🔥 确保 context 的 dtype 与 out_proj 权重匹配
        # 如果我们转换过 dtype（为了 FlashBias），需要转换回去
        if context.dtype != self.out_proj.weight.dtype:
            context = context.to(self.out_proj.weight.dtype)
        
        # 输出投影
        return self.out_proj(context), None
    
    def _flashbias_attention(self, q, k, v, attn_bias):
        """
        Triton-based FlashBias 实现
        FlashBias 需要输入格式: [batch, seqlen, nheads, headdim]
        要求：所有输入必须是 fp16 或 bf16
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        _, _, seq_len_k, _ = k.shape
        
        # 🔥 确保输入是 fp16/bf16（FlashBias 要求）
        if q.dtype not in [torch.float16, torch.bfloat16]:
            raise TypeError(f"FlashBias requires fp16/bf16, but got {q.dtype}")
        
        # 🔥 将 attn_bias 转换为 q_bias 和 k_bias（通过 SVD 低秩分解）
        q_bias, k_bias = self._convert_attn_bias_to_qk_bias(attn_bias)
        # SVD 输出: q_bias [batch, num_heads, seq_len_q, rank]
        #          k_bias [batch, num_heads, seq_len_k, rank]
        
        # 🔥 转换为 FlashBias 期望的格式: [batch, seqlen, nheads, headdim/rank]
        q_flash = q.transpose(1, 2)  # [batch, seq_len_q, num_heads, head_dim]
        k_flash = k.transpose(1, 2)  # [batch, seq_len_k, num_heads, head_dim]
        v_flash = v.transpose(1, 2)  # [batch, seq_len_k, num_heads, head_dim]
        q_bias_flash = q_bias.transpose(1, 2)  # [batch, seq_len_q, num_heads, rank]
        k_bias_flash = k_bias.transpose(1, 2)  # [batch, seq_len_k, num_heads, rank]
        
        # 🔥 调用 FlashBias（Triton 实现）
        # 注意：flash_bias_func 不接受关键字参数，必须按位置传递
        context = flash_bias_func(
            q_flash,          # q
            k_flash,          # k
            v_flash,          # v
            q_bias_flash,     # q_bias
            k_bias_flash,     # k_bias
            None,             # mask
            False,            # causal
            1.0 / math.sqrt(head_dim)  # softmax_scale
        )
        # FlashBias 输出: [batch, seq_len_q, num_heads, head_dim]
        
        # 🔥 转回标准格式 [batch, num_heads, seq_len_q, head_dim]
        context = context.transpose(1, 2)
        
        # 转回 [batch, seq_len_q, embed_dims]
        context = context.contiguous().view(batch_size, seq_len_q, -1)
        return context
    
    def _pytorch_sdpa_attention(self, q, k, v, attn_bias=None):
        """
        PyTorch-SDPA-based FlashBias 实现（GitHub官方方法2）
        使用 concat([q*scale, q_bias], [k, k_bias]) 的方式
        要求：concat 后的维度能被8整除，才能激活 FlashAttention 后端
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        _, _, seq_len_k, _ = k.shape
        
        if attn_bias is not None:
            # 🔥 将 attn_bias 通过 SVD 分解为 q_bias 和 k_bias
            # SVD输出: q_bias [batch, num_heads, seq_len_q, rank]
            #         k_bias [batch, num_heads, seq_len_k, rank]
            q_bias, k_bias = self._convert_attn_bias_to_qk_bias(attn_bias)
            
            # 🔥 PyTorch-SDPA-based FlashBias (GitHub官方方法)
            # 要求: concat[q, q_bias] 的最后一维能被8整除
            rank = q_bias.shape[-1]
            total_dim = head_dim + rank
            
            # 检查是否需要padding
            if total_dim % 8 != 0:
                pad_size = 8 - (total_dim % 8)
                # 对 q_bias 和 k_bias 进行padding
                q_bias = torch.cat([q_bias, torch.zeros(batch_size, num_heads, seq_len_q, pad_size, device=q_bias.device, dtype=q_bias.dtype)], dim=-1)
                k_bias = torch.cat([k_bias, torch.zeros(batch_size, num_heads, seq_len_k, pad_size, device=k_bias.device, dtype=k_bias.dtype)], dim=-1)
            
            # 计算 softmax_scale
            softmax_scale = 1.0 / math.sqrt(head_dim)
            
            # 🔥 按照 FlashBias 官方方式拼接: concat([q*scale, q_bias], [k, k_bias])
            q_concat = torch.cat([q * softmax_scale, q_bias], dim=-1)  # [batch, num_heads, seq_len_q, head_dim+rank]
            k_concat = torch.cat([k, k_bias], dim=-1)                   # [batch, num_heads, seq_len_k, head_dim+rank]
            
            # 使用 PyTorch SDPA（自动使用 FlashAttention）
            context = torch.nn.functional.scaled_dot_product_attention(
                query=q_concat,
                key=k_concat,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                scale=1.0,  # 已经在 q 上乘过 scale 了
                is_causal=False
            )
            # 输出: [batch, num_heads, seq_len_q, head_dim]
            
            # 转回 [batch, seq_len_q, embed_dims]
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
            return context
        else:
            # 没有偏置，使用标准 SDPA
            # q, k, v 已经是 [batch, num_heads, seq_len, head_dim] 格式
            context = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                scale=1.0 / math.sqrt(head_dim),
                is_causal=False
            )
            # 输出: [batch, num_heads, seq_len_q, head_dim]
            
            # 转回 [batch, seq_len_q, embed_dims]
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
            return context
    
    def _standard_attention(self, q, k, v, attn_bias=None):
        """
        高效注意力计算（使用 PyTorch SDPA）
        PyTorch 2.0+ 的 SDPA 会自动使用 FlashAttention（如果可用）
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        _, _, seq_len_k, _ = k.shape
        
        # 🔥 使用 PyTorch 2.0+ 的 scaled_dot_product_attention
        # 输入格式: [batch, num_heads, seq_len, head_dim]
        # q, k, v 已经是这个格式，不需要转置
        
        # 处理 attention_bias
        attn_mask_sdpa = None
        if attn_bias is not None:
            # SDPA 期望 attn_mask 的格式: [batch, num_heads, seq_len_q, seq_len_k]
            # 或 [batch, 1, seq_len_q, seq_len_k]（会broadcast到所有heads）
            if attn_bias.dim() == 3:
                # [batch, seq_len_q, seq_len_k] → [batch, 1, seq_len_q, seq_len_k]
                attn_mask_sdpa = attn_bias.unsqueeze(1)
            elif attn_bias.dim() == 4:
                # 已经是正确格式 [batch, num_heads, seq_len_q, seq_len_k]
                attn_mask_sdpa = attn_bias
            else:
                # 其他情况，扩展到4维
                attn_mask_sdpa = attn_bias.unsqueeze(0).unsqueeze(0)
            
            # 🔥 关键修复：确保 attn_mask 的 dtype 与 query 匹配
            # PyTorch SDPA 要求 attn_mask 的 dtype 要么是 bool，要么与 query 相同
            if attn_mask_sdpa.dtype != q.dtype:
                attn_mask_sdpa = attn_mask_sdpa.to(q.dtype)
        
        # 使用 PyTorch SDPA（自动使用 FlashAttention）
        # 输入格式: [batch, num_heads, seq_len, head_dim]
        context = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask_sdpa,  # 🔥 传递 attention_bias
            dropout_p=self.dropout if self.training else 0.0,
            scale=None,  # 使用默认 scale (1/sqrt(head_dim))
            is_causal=False
        )
        # SDPA 输出: [batch, num_heads, seq_len_q, head_dim]
        
        # 转回 [batch, seq_len_q, embed_dims]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        
        return context
    
    def _convert_attn_bias_to_qk_bias(self, attn_bias, rank=None):
        """
        将 attn_bias 转换为 q_bias 和 k_bias
        
        Args:
            attn_bias: [batch, num_heads, seq_len, seq_len] 注意力偏置矩阵
                      或 [batch, seq_len, seq_len] 简化格式（会自动扩展到多头）
            rank: 低秩近似的秩，如果为None则自动选择
            
        Returns:
            q_bias: [batch, num_heads, seq_len, rank] 查询偏置
            k_bias: [batch, num_heads, rank, seq_len] 键偏置
        """
        # 🔥 SVD 不支持 FP16，需要转为 FP32
        original_dtype = attn_bias.dtype
        if attn_bias.dtype == torch.float16:
            attn_bias = attn_bias.float()
        
        # 🔥 处理 3 维输入（自动扩展到多头）
        if attn_bias.dim() == 3:
            # [batch, seq_len_q, seq_len_k] → [batch, num_heads, seq_len_q, seq_len_k]
            attn_bias = attn_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        batch_size, num_heads, seq_len_q, seq_len_k = attn_bias.shape
        
        # 🔥 使用固定的小 rank 以节省显存（避免 OOM）
        # 原来的自动计算 rank 非常占显存，且在大矩阵上会导致 OOM
        if rank is None:
            # 使用固定的小 rank，在性能和显存之间取平衡
            # rank=8 通常足够捕捉主要的注意力模式
            rank = min(8, min(seq_len_q, seq_len_k) // 4)
        
        # 🔥 在 GPU 上进行 SVD 分解（FlashBias 会节省显存）
        device = attn_bias.device
        print(f"🔄 开始 SVD 分解 (GPU)：batch={batch_size}, heads={num_heads}, seq_q={seq_len_q}, seq_k={seq_len_k}, rank={rank}")
        
        # 🔥 批量 SVD：将 batch 和 heads 维度合并
        # [batch, num_heads, seq_q, seq_k] -> [batch*num_heads, seq_q, seq_k]
        attn_bias_flat = attn_bias.reshape(batch_size * num_heads, seq_len_q, seq_len_k)
        
        # 批量 SVD 分解（在 GPU 上）
        U, S, V = torch.svd(attn_bias_flat)
        
        # 选择前 rank 个奇异值
        U_trunc = U[:, :, :rank]  # [batch*num_heads, seq_len_q, rank]
        S_trunc = S[:, :rank]     # [batch*num_heads, rank]
        V_trunc = V[:, :, :rank]  # [batch*num_heads, seq_len_k, rank]
        
        # 🔥 重构偏置：attn_bias = q_bias @ k_bias.T
        # 使用 sqrt(S) 来分配奇异值的权重
        sqrt_S = torch.sqrt(S_trunc)  # [batch*num_heads, rank]
        q_bias = U_trunc * sqrt_S.unsqueeze(1)  # [batch*num_heads, seq_len_q, rank]
        k_bias = V_trunc * sqrt_S.unsqueeze(1)  # [batch*num_heads, seq_len_k, rank]
        
        # 🔥 重新组织为 FlashBias 期望的格式
        # [batch*num_heads, seq_len, rank] -> [batch, num_heads, seq_len, rank]
        q_bias = q_bias.view(batch_size, num_heads, seq_len_q, rank)
        k_bias = k_bias.view(batch_size, num_heads, seq_len_k, rank)
        
        print(f"✅ SVD 分解完成")
        
        # 🔥 转回原始 dtype
        if original_dtype == torch.float16:
            q_bias = q_bias.half()
            k_bias = k_bias.half()
        
        return q_bias, k_bias


# 🔥 为了兼容性，保留原有的 FlashMHA 类名
class FlashMHA(FlashBiasAttention):
    """
    为了兼容性保留的别名
    """
    pass


# 🔥 为了兼容性，保留原有的 FlashAttention 类名  
class FlashAttention(FlashBiasAttention):
    """
    为了兼容性保留的别名
    """
    pass