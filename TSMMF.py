'''

This script realizes the multimodal temporal-spatial fusion of EEG and fNIRS (TSMMF).

'''


import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange

from DualConvLayer import SpatialConvLayer, TemporalConvLayer

# 位置信息通过向量加法的形式修改了原始的特征向量 使其携带了位置的语义
# emb_size超参数 定义了模型内部表示各种输入单元的特征向量的长度
# nn.Parameter 自定义位置嵌入 自定义分类Token(CLS Token) 可学习的参数或缩放因子 自定义注意力机制的权重
# forward 将可学习的位置嵌入（self.pos_emb）叠加到输入的特征（x）上
class PositionalEmbedding(nn.Module):
    def __init__(self, channels, emb_size, device):
        super().__init__()
        self.channels = channels + 1
        self.pos_emb = nn.Parameter(torch.randn(size=(1, self.channels, emb_size), dtype=torch.float32, device=device),
                                    requires_grad=True)

    def forward(self, x):
        x = self.pos_emb + x
        return x

# nn.Embedding 将离散的整数索引（在这里是模态类型 0 或 1）映射到连续的浮点向量
# 为拼接的多模态输入注入模态类型信息 可学习的嵌入向量加入原始输入
class ModalityTypeEmbedding(nn.Module):
    def __init__(self, emb_size, token_type_idx=1):
        super().__init__()
        self.token_type_idx = token_type_idx
        self.type_embedding_layer = nn.Embedding(2, emb_size)

    def forward(self, x, mask):
        # x.shape = B, 1 + eeg_tokens + 1 + nirs_tokens, emb_size
        # mask.shape = [1 + eeg_tokens, 1 + nirs_tokens]
        b, _, emb_size = x.shape
        modality_type_emb = torch.ones(b, mask[0] + mask[1], dtype=torch.long, device=x.device)
        modality_type_emb[:, mask[0]::] = 0
        type_emb = self.type_embedding_layer(modality_type_emb)
        x = x + type_emb
        return x


# 输入特征向量增加可学习参数变为 （batch_size, sequence_length+1, emb_size)
class AddClsToken(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size), requires_grad=True)

    def forward(self, x):
        self.cls_token = self.cls_token.to(x.device)
        x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        return x


# Q、K、V 通常都是从同一个原始输入序列通过不同的Linear层线性投影得到的
class MultiHeadAttention(nn.Module):
    # 初始化多头注意力机制所需的所有组件和参数
    def __init__(self, query_size, key_size, value_size, emb_size, num_heads, dropout, bias=False):
        super().__init__()
        self.emb_size = emb_size
        self.proj_dim = self.emb_size
        self.num_heads = num_heads
        self.queries = nn.Linear(query_size, self.proj_dim, bias=bias)
        self.keys = nn.Linear(key_size, self.proj_dim, bias=bias)
        self.values = nn.Linear(value_size, self.proj_dim, bias=bias)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(self.proj_dim, emb_size, bias=bias),
            nn.Dropout(dropout)
        )
        self.attention_weights = None
    # 定义了多头注意力机制实际的计算流程，也就是如何从输入 query, key, value 得到最终的注意力输出
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(query), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(key), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(value), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        self.attention_weights = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(self.attention_weights)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


# 自注意力 Q K V来自同一源 交叉注意力 （Q来自一个源 K V来自另一个源)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, emb_size, num_heads, dropout, bias=False):
        super().__init__()
        self.proj_dim = emb_size
        self.num_heads = num_heads
        self.queries = nn.Linear(query_size, self.proj_dim, bias=bias)
        self.keys = nn.Linear(key_size, self.proj_dim, bias=bias)
        self.values = nn.Linear(value_size, self.proj_dim, bias=bias)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(self.proj_dim, emb_size, bias=bias),
            nn.Dropout(dropout)
        )
        self.attention_weights = None
        self.scaling = emb_size ** (1 / 2)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(query), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(key), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(value), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        self.attention_weights = F.softmax(energy / self.scaling, dim=-1)
        att = self.att_drop(self.attention_weights)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

# Transformer前馈网络 线性层 GELU激活函数 线性层
class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion=2, dropout=0.5):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.feed_forward(x)

        return x


# 层归一化 多头注意力 残差相加 层归一化 前馈网络 残差相加
class SelfEncoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, emb_size, num_heads=4, forward_expansion=2, dropout=0.5):
        super(SelfEncoderBlock, self).__init__()

        self.attention = MultiHeadSelfAttention(query_size, key_size, value_size, emb_size, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(emb_size, expansion=forward_expansion, dropout=dropout)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x):
        # x是主导模态
        res = x
        x = self.norm1(x)
        y = self.attention(x, x, x)
        y = self.dropout_attn(y)
        y = y + res

        res = y
        y2 = self.norm2(y)
        y2 = self.feed_forward(y2)
        y2 = self.dropout_ffn(y2)
        y2 = y2 + res

        return y2


# 输入 -> 层归一化 -> 多头注意力模块 -> 残差相加 -> 层归一化 -> 前馈网络模块 -> 残差相加
class CrossEncoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, emb_size, num_heads=4, forward_expansion=2, dropout=0.7):
        super(CrossEncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(query_size, key_size, value_size, emb_size, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(emb_size, expansion=forward_expansion, dropout=dropout)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size)

        self.dropout_attn = nn.Dropout(dropout) # 用于注意力输出
        self.dropout_ffn = nn.Dropout(dropout)  # 用于前馈网络输出

    def forward(self, x, y):
        # x是主导模态
        res = x
        x, y = self.norm1(x), self.norm2(y)
        y1 = self.attention(x, y, y)
        y1 = self.dropout_attn(y1)
        y1 = y1 + res

        res = y1
        y2 = self.norm3(y1)
        y2 = self.feed_forward(y2)
        y2 = self.dropout_ffn(y2)
        y2 = y2 + res

        return y2


# TransformerCrossEncoder是一个由多个CrossEncoderBlock堆叠而成的整体结构
class TransformerCrossEncoder(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, dropout,
                 device):
        super(TransformerCrossEncoder, self).__init__()

        self.blks = nn.Sequential()
        self.attention_weights = [None] * depth

        for i in range(depth):
            self.blks.add_module("block" + str(i),
                                 CrossEncoderBlock(query_size, key_size, value_size, emb_size, num_heads, expansion,
                                                   dropout))

    def forward(self, x, y):
        for i, blk in enumerate(self.blks):
            x = blk(x, y)
            self.attention_weights[i] = blk.attention.attention_weights
        return x

    @property
    def cross_attention_weights(self):
        return self.attention_weights


# TransformerCatEncoder是一个将多个SelfEncoderBlock堆叠起来的编码器 接受两个独立的输入序列x和y 核心在于处理和融合这两个序列的信息
class TransformerCatEncoder(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, dropout,
                 device):
        super(TransformerCatEncoder, self).__init__()
        self.modality_embedding = ModalityTypeEmbedding(emb_size)
        self.blks = nn.Sequential()
        self.attention_weights = [None] * depth

        for i in range(depth):
            self.blks.add_module("block" + str(i),
                                 SelfEncoderBlock(query_size, key_size, value_size, emb_size, num_heads, expansion,
                                                  dropout))

    def forward(self, x, y, mask=None):
        context = torch.cat([x, y], dim=1)
        context = self.modality_embedding(context, mask)
        for i, blk in enumerate(self.blks):
            context = blk(context)
            self.attention_weights[i] = blk.attention.attention_weights
        return context

    @property
    def self_attention_weights(self):
        return self.attention_weights


# 接收单个输入序列x 对这一个序列进行深度编码和特征提取
class TransformerSelfEncoder(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, dropout,
                 device):
        super(TransformerSelfEncoder, self).__init__()
        self.blks = nn.Sequential()
        self.attention_weights = [None] * depth
        self.add_token = AddClsToken(emb_size)
        self.positional_embedding = PositionalEmbedding(channels, emb_size, device)
        for i in range(depth):
            self.blks.add_module("block" + str(i),
                                 SelfEncoderBlock(query_size, key_size, value_size, emb_size, num_heads, expansion,
                                                  dropout))

    def forward(self, x, mask=None):
        x = self.add_token(x)
        x = self.positional_embedding(x)
        for i, blk in enumerate(self.blks):
            x = blk(x)
            self.attention_weights[i] = blk.attention.attention_weights
        return x

    @property
    def self_attention_weights(self):
        return self.attention_weights


class Transformer(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, device,
                 self_dropout, cross_dropout):
        super().__init__()
        self.eeg_nirs_temporal_spatial_attention_weights = None
        self.eeg_temporal_spatial_attention_weights = None
        self.eeg_temporal_attention_weights = None
        self.eeg_spatial_attention_weights = None
        self.nirs_spatial_attention_weights = None
        self.temporal_mask = [channels[0] + 1, channels[2] + 1]
        self.spatial_mask = [channels[1] + 1, channels[3] + 1]


        self.eeg_spatial_encoder = TransformerSelfEncoder(depth[0], query_size, key_size, value_size, emb_size, num_heads,
                                                          64, expansion, self_dropout, device)

        self.nirs_spatial_encoder = TransformerSelfEncoder(depth[0], query_size, key_size, value_size, emb_size,
                                                           num_heads, 64, expansion, cross_dropout, device)

        self.eeg_temporal_encoder = TransformerSelfEncoder(depth[0], query_size, key_size, value_size, emb_size,
                                                           num_heads, channels[0], expansion, cross_dropout, device)

        self.nirs_temporal_encoder = TransformerSelfEncoder(depth[0], query_size, key_size, value_size, emb_size,
                                                            num_heads, channels[2], expansion, cross_dropout, device)

        self.spatial_context = TransformerCatEncoder(1, query_size, key_size, value_size, emb_size,
                                             num_heads, None, expansion, cross_dropout, device)

        self.temporal_context = TransformerCatEncoder(1, query_size, key_size, value_size, emb_size,
                                                     num_heads, None, expansion, cross_dropout, device)

        self.eeg_temporal_cross_encoder = TransformerCrossEncoder(depth[1], query_size, key_size, value_size, emb_size,
                                                         num_heads, None, expansion, cross_dropout, device)
        self.nirs_temporal_cross_encoder = TransformerCrossEncoder(depth[1], query_size, key_size, value_size, emb_size,
                                                          num_heads, None, expansion, cross_dropout, device)

        self.eeg_spatial_cross_encoder = TransformerCrossEncoder(depth[1], query_size, key_size, value_size, emb_size,
                                                                 num_heads, None, expansion, cross_dropout, device)
        self.nirs_spatial_cross_encoder = TransformerCrossEncoder(depth[1], query_size, key_size, value_size, emb_size,
                                                                  num_heads, None, expansion, cross_dropout, device)

    def forward(self, temporal_eeg, temporal_nirs, spatial_eeg, spatial_nirs):

        eeg_spatial_outputs = self.eeg_spatial_encoder(spatial_eeg)
        nirs_spatial_outputs = self.nirs_spatial_encoder(spatial_nirs)

        eeg_temporal_outputs = self.eeg_temporal_encoder(temporal_eeg)
        nirs_temporal_outputs = self.nirs_temporal_encoder(temporal_nirs)

        spatial_context = self.spatial_context(eeg_spatial_outputs, nirs_spatial_outputs, self.spatial_mask)
        temporal_context = self.temporal_context(eeg_temporal_outputs, nirs_temporal_outputs, self.temporal_mask)

        eeg_temporal_cross_outputs = self.eeg_temporal_cross_encoder(eeg_temporal_outputs, temporal_context)
        nirs_temporal_cross_outputs = self.nirs_temporal_cross_encoder(nirs_temporal_outputs, temporal_context)

        eeg_spatial_cross_outputs = self.eeg_spatial_cross_encoder(eeg_spatial_outputs, spatial_context)
        nirs_spatial_cross_outputs = self.nirs_spatial_cross_encoder(nirs_spatial_outputs, spatial_context)

        self.eeg_spatial_attention_weights = self.eeg_spatial_encoder.self_attention_weights
        self.nirs_spatial_attention_weights = self.nirs_spatial_encoder.self_attention_weights

        self.eeg_nirs_spatial_attention_weights = self.eeg_spatial_cross_encoder.cross_attention_weights
        self.nirs_eeg_spatial_attention_weights = self.nirs_spatial_cross_encoder.cross_attention_weights

        return [eeg_temporal_outputs[:, 0], nirs_temporal_outputs[:, 0],
                [eeg_temporal_cross_outputs[:, 0], nirs_temporal_cross_outputs[:, 0]],
                eeg_spatial_outputs[:, 0], nirs_spatial_outputs[:, 0],
                [eeg_spatial_cross_outputs[:, 0], nirs_spatial_cross_outputs[:, 0]]]

    @property
    def get_spatial_self_attention_weights(self):
        return [self.eeg_spatial_attention_weights, self.nirs_spatial_attention_weights]

    @property
    def get_spatial_cross_attention_weights(self):
        return [self.eeg_nirs_spatial_attention_weights, self.nirs_eeg_spatial_attention_weights]


class AttentionFusion(nn.Module):
    def __init__(self, emb_size):
        super(AttentionFusion, self).__init__()
        self.weight = nn.Parameter(torch.randn(emb_size, 1), requires_grad=True)
        self.softmax = nn.Softmax(-1)
        self.alpha = None

    def forward(self, out):
        o = torch.cat([i @ self.weight for i in out], dim=-1)
        self.alpha = self.softmax(o)
        outputs = sum([i * self.alpha[:, index].unsqueeze(1) for index, i in enumerate(out)])
        return outputs

# 多层次特征融合 AttentionFusion融合EEG时空特征、NIRS时空特征、交叉模态特征
# 通过可学习的权重最终分类输出
class ClassificationHead(nn.Module):
    def __init__(self, num_classes, emb_size, dropout):
        super(ClassificationHead, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.attention_weight_sum_fusion = AttentionFusion(emb_size)
        self.attention_weight_sum_eeg = AttentionFusion(emb_size)
        self.attention_weight_sum_nirs = AttentionFusion(emb_size)
        self.eeg = nn.Sequential(
            nn.Linear(emb_size * 1, num_classes),
        )
        self.nirs = nn.Sequential(
            nn.Linear(emb_size * 1, num_classes),
        )
        self.fusion = nn.Sequential(
            nn.Linear(emb_size * 1, num_classes)
        )
        self.w = nn.Parameter(torch.Tensor([1., 0.01, 0.01]), requires_grad=True)

    def forward(self, out):
        eeg_temporal, nirs_temporal, temporal_cross, eeg_spatial, nirs_spatial, spatial_cross = out
        cross = temporal_cross + spatial_cross
        cross_outputs = self.attention_weight_sum_fusion(cross)
        eeg_outputs = self.attention_weight_sum_eeg([eeg_temporal, eeg_spatial])
        nirs_outputs = self.attention_weight_sum_nirs([nirs_temporal, nirs_spatial])

        eeg_outputs = self.dropout_layer(eeg_outputs)
        nirs_outputs = self.dropout_layer(nirs_outputs)
        cross_outputs = self.dropout_layer(cross_outputs)

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))

        out = self.eeg(eeg_outputs) * w1 + self.nirs(nirs_outputs) * w2 + self.fusion(cross_outputs) * w3

        return out


class HybridTransformer(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, expansion, conv_dropout,
                 self_dropout, cross_dropout, cls_dropout, num_classes, device):
        super().__init__()
        self.spatial_conv_layer = SpatialConvLayer(emb_size, conv_dropout)
        self.temporal_conv_layer = TemporalConvLayer(emb_size, conv_dropout)

        with torch.no_grad():
            eeg, nirs = torch.randn(1, 64, 2000), torch.randn(1, 64, 2000) # <-- 这一行需要修改
            eeg_temporal_token, nirs_temporal_token = self.temporal_conv_layer(eeg, nirs)
            eeg_spatial_token, nirs_spatial_token = self.spatial_conv_layer(eeg, nirs)
            channels = [eeg_temporal_token.shape[-1], eeg_spatial_token.shape[-2], nirs_temporal_token.shape[-1], nirs_spatial_token.shape[-2]]

        self.transformer = Transformer(depth, query_size, key_size, value_size, emb_size, num_heads, channels,
                                       expansion, device, self_dropout, cross_dropout)
        self.classify = ClassificationHead(num_classes, emb_size, cls_dropout)

    def forward(self, eeg, nirs):
        spatial_eeg, spatial_nirs = self.spatial_conv_layer(eeg, nirs)
        temporal_eeg, temporal_nirs = self.temporal_conv_layer(eeg, nirs)
        temporal_eeg, temporal_nirs = temporal_eeg.squeeze(-2).permute(0, 2, 1), temporal_nirs.squeeze(-2).permute(0, 2, 1)
        spatial_eeg, spatial_nirs = spatial_eeg.squeeze(-1).permute(0, 2, 1), spatial_nirs.squeeze(-1).permute(0, 2, 1)
        out1 = self.transformer(temporal_eeg, temporal_nirs, spatial_eeg, spatial_nirs)
        outputs = self.classify(out1)

        return outputs
