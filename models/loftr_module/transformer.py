import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention
import math
from einops import rearrange
import torch.nn.functional as F


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 # 【修复】为 Pre-Norm 添加 eps (epsilon) 以提高稳定性
                 layer_norm_eps=1e-5):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        # 【修复】修改 MLP 以适应 Pre-Norm (不再需要 cat)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),  # 原始: d_model * 2
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),  # 原始: d_model * 2 -> d_model
        )

        # norm and dropout
        # 【修复】使用传入的 eps
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        【修复】: 重构为 Pre-Norm 架构 (Norm -> Attention -> Add -> Norm -> MLP -> Add)

        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)

        # --- 1. Multi-Head Attention (Pre-Norm) ---
        # 归一化 *之前*
        x_norm = self.norm1(x)
        # (在 'self' attention 中, source == x, source_norm == x_norm)
        # (在 'cross' attention 中, source != x, 我们需要归一化 source)
        source_norm = self.norm1(source)

        query, key, value = x_norm, source_norm, source_norm

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]

        # 第一个残差连接
        x_with_attn = x + message

        # --- 2. Feed-Forward Network (Pre-Norm) ---
        # 归一化 *之前*
        x_norm_2 = self.norm2(x_with_attn)

        # feed-forward network
        message_mlp = self.mlp(x_norm_2)

        # 第二个残差连接
        x_final = x_with_attn + message_mlp

        return x_final


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']

        # 【修复】将稳定性修复 (eps) 传递给 EncoderLayer
        layer_norm_eps = config.get('layer_norm_eps', 1e-5)
        encoder_layer = LoFTREncoderLayer(
            config['d_model'],
            config['nhead'],
            config['attention'],
            layer_norm_eps=layer_norm_eps
        )
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1


import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention  # 假设此文件存在
import math
from einops import rearrange
import torch.nn.functional as F

# import torch.nn.utils.rnn as rnn_utils # 优化后不再需要 rnn_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import copy
import math  # 确保导入 math (用于 UNFOLD)


# 假设 LoFTREncoderLayer 在别处定义
# from .loftr_module import LoFTREncoderLayer
class LoFunction_Placeholder(nn.Module):  # 占位符
    def __init__(self, d_model, nhead, attention_type, **kwargs):  # (添加 **kwargs 以接受 eps)
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, feat0, feat1, mask0, mask1):
        # 这是一个简化的占位符，实际的 LoFTR 层更复杂
        if feat1 is not None:
            return self.attn(feat0.transpose(0, 1), feat1.transpose(0, 1), feat1.transpose(0, 1))[0].transpose(0, 1)
        else:
            return self.attn(feat0.transpose(0, 1), feat0.transpose(0, 1), feat0.transpose(0, 1))[0].transpose(0, 1)


# 【修复】删除占位符覆盖，我们现在使用上面修复过的真实 LoFTREncoderLayer
# LoFTREncoderLayer = LoFunction_Placeholder  # <-- 已删除

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import copy
import math


# (假设 LoFTREncoderLayer 占位符已定义)
# (为简洁起见，省略 LoFTREncoderLayer 占位符)
# class LoFunction_Placeholder(nn.Module):  # 占位符 (已在上方定义)
#     ...

# 【修复】删除占位符覆盖，我们现在使用上面修复过的真实 LoFTREncoderLayer
# LoFTREncoderLayer = LoFunction_Placeholder  # <-- 已删除


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class TopicFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names_t']
        self.n_topics = config['n_topics']

        # Transformer Encoder layers
        layer_norm_eps = config.get('layer_norm_eps', 1e-5)
        encoder_layer = LoFTREncoderLayer(
            self.d_model,
            self.nhead,
            config['attention'],
            layer_norm_eps=layer_norm_eps
        )
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])

        # seed tokens
        self.seed_tokens = nn.Parameter(torch.randn(self.n_topics, self.d_model))
        self.register_parameter('seed_tokens', self.seed_tokens)
        self.topic_drop = nn.Dropout1d(p=0.1)

        # LayerNorm
        self.norm_feat = nn.LayerNorm(self.d_model, eps=layer_norm_eps)

        # 可学习温度
        self.topic_temperature = nn.Parameter(torch.tensor(config.get('topic_temperature', 10.0)))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, topic0, topic1, mask0=None, mask1=None, smooth=True):
        N, L, S, C, K = feat0.shape[0], feat0.shape[1], feat1.shape[1], feat0.shape[2], self.n_topics

        # 1. 初始化 seeds
        seeds = self.seed_tokens.unsqueeze(0).repeat(N, 1, 1)  # [N, K, C]
        seeds = self.topic_drop(seeds)

        # 2. 计算主题相似度矩阵 top_matrix
        # top_matrix = torch.einsum("nlc,nsc->nls", topic0, topic1) / (C ** 0.5)

        # 双向 softmax + 可选 Sinkhorn
        # top_matrix_row = F.softmax(top_matrix, dim=2)
        # top_matrix_col = F.softmax(top_matrix, dim=1)
        # top_matrix_norm = top_matrix_row * top_matrix_col  # [N, L, S]

        # 3. soft attention warp (替代原来的 argmax warp)
        # 每个 patch 在 topic1 上按概率加权
        # topic1_warped = torch.einsum("nls,nsc->nlc", top_matrix_norm, topic1)
        # topic = (topic0 + topic1_warped) / 2  # [N, L, K]

        # 4. feature ↔ seed 交互
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'seed':
                # 用 residual/加权融合避免过度平滑
                seeds = layer(seeds, feat0, None, mask0)
                # seeds_new2 = layer(seeds_new, topic, None, mask0)
                # seeds = 0.5 * (seeds + seeds_new2)
            elif name == 'feat':
                feat0 = layer(feat0, seeds, mask0, None)
                feat1 = layer(feat1, seeds, mask1, None)
                # feat0 = 0.5 * (feat0 + feat0_new)
                # feat1 = 0.5 * (feat1 + feat1_new)

        # 5. 归一化
        feat0 = self.norm_feat(feat0)
        feat1 = self.norm_feat(feat1)
        seeds = self.norm_feat(seeds)

        # 6. 计算主题概率矩阵
        # 每个 patch 与 seed 点乘，不再 concat feat0+feat1
        dmatrix0 = torch.einsum("nlc,nkc->nlk", feat0, seeds) / self.topic_temperature
        dmatrix1 = torch.einsum("nsc,nkc->nsk", feat1, seeds) / self.topic_temperature

        prob_topics0 = F.softmax(dmatrix0, dim=-1)
        prob_topics1 = F.softmax(dmatrix1, dim=-1)

        # if smooth:
        #     # 可选 2D 平滑
        #     H0 = W0 = int(L**0.5)
        #     if H0*W0 != L: H0 = W0 = int(L**0.5)
        #     prob_topics0_2d = prob_topics0.permute(0,2,1).reshape(N, K, H0, W0)
        #     prob_topics0_2d = F.avg_pool2d(prob_topics0_2d, 3, 1, 1)
        #     prob_topics0 = prob_topics0_2d.flatten(2).permute(0,2,1)
        #
        #     H1 = W1 = int(S**0.5)
        #     if H1*W1 != S: H1 = W1 = int(S**0.5)
        #     prob_topics1_2d = prob_topics1.permute(0,2,1).reshape(N, K, H1, W1)
        #     prob_topics1_2d = F.avg_pool2d(prob_topics1_2d, 3, 1, 1)
        #     prob_topics1 = prob_topics1_2d.flatten(2).permute(0,2,1)

        topic_matrix_match = {"img0": prob_topics0, "img1": prob_topics1}

        return feat0, feat1, topic_matrix_match


class FineNetwork(nn.Module):

    def __init__(self, config, add_detector=True):
        super(FineNetwork, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']

        # 【修复】将稳定性修复 (eps) 传递给 EncoderLayer
        layer_norm_eps = config.get('layer_norm_eps', 1e-5)
        encoder_layer = LoFTREncoderLayer(
            config['d_model'],
            config['nhead'],
            config['attention'],
            layer_norm_eps=layer_norm_eps
        )
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        # --- 修复结束 ---

        self.detector = None
        if add_detector:
            # 假设 detector 也使用 LoFTREncoderLayer
            self.detector = nn.Sequential(
                LoFTREncoderLayer(config["d_model"], config['nhead'], config['attention'],
                                  layer_norm_eps=layer_norm_eps),  # 【修复】
                nn.Linear(self.d_model, 1)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.shape[2], "the feature number of src and transformer must be equal"

        # --- 修复: 使用 LocalFeatureTransformer 的标准 self/cross 逻辑 ---
        for layer, name in zip(self.encoder_layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
        # --- 修复结束 ---

        score_map0 = None
        if self.detector is not None:
            # detector (self-attention)
            feat0_detect = self.detector[0](feat0, feat0, mask0, mask0)
            score_map0 = self.detector[1](feat0_detect).squeeze(-1)

        return feat0, feat1, score_map0


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FeatureFusion(nn.Module):

    def __init__(self, config):
        super(FeatureFusion, self).__init__()
        layer_norm_eps = config.get('layer_norm_eps', 1e-5)
        self.config = config
        self.d_model = config['d_model']
        self.d_model_fusion = config['d_model']
        self.nhead = config['nhead']
        self.avgpool = nn.AdaptiveAvgPool1d(160)
        self.avgpool_1 = nn.AdaptiveAvgPool1d(96)
        # 您的 config 传入 d_model=256，这与 160+96=256 匹配，是正确的
        self.norm_feat = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.head = 4
        self.thr = config.get('thr', 0.1)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_s0, feat_s1, data=None):
        N, L, C = feat_s0.shape
        S = feat_s1.shape[1]
        H0, W0 = int(math.sqrt(L)), int(math.sqrt(L))
        H1, W1 = int(math.sqrt(S)), int(math.sqrt(S))
        if H0 * W0 != L:
            print(f"Warning (FeatureFusion): L ({L}) is not a perfect square. Using floored sqrt.")
            H0 = W0 = int(L ** 0.5)
        if H1 * W1 != S:
            print(f"Warning (FeatureFusion): S ({S}) is not a perfect square. Using floored sqrt.")
            H1 = W1 = int(S ** 0.5)

        assert self.d_model_fusion == C, "the feature number of src and transformer must be equal"

        conf_matrix = torch.einsum("nlc,nsc->nls", feat_s0, feat_s1) / self.d_model ** .5
        data["conf_matrix"] = conf_matrix

        conf_matrix_softmax = F.softmax(conf_matrix, 1) * F.softmax(conf_matrix, 2)
        conf_mask = conf_matrix_softmax > self.thr
        conf_mask = conf_mask * (conf_matrix_softmax == conf_matrix_softmax.max(dim=2, keepdim=True)[0]) \
                    * (conf_matrix_softmax == conf_matrix_softmax.max(dim=1, keepdim=True)[0])
        conf_mask = conf_mask.float()  # [N, L, S]

        feat_s0_2d = torch.reshape(feat_s0, (N, C, H0, W0))
        feat_s1_2d = torch.reshape(feat_s1, (N, C, H1, W1))

        feat_s0_unfold = F.unfold(feat_s0_2d, kernel_size=(3, 3), stride=1, padding=1)
        feat_s0_unfold = rearrange(feat_s0_unfold, 'n (c ww) l -> n l ww c', ww=3 ** 2)
        feat_s0_sem = torch.einsum("nlc,nlwc->nlw", feat_s0, feat_s0_unfold) / feat_s0_unfold.shape[2]

        feat_s1_unfold = F.unfold(feat_s1_2d, kernel_size=(3, 3), stride=1, padding=1)
        feat_s1_unfold = rearrange(feat_s1_unfold, 'n (c ww) l -> n l ww c', ww=3 ** 2)
        feat_s1_sem = torch.einsum("nlc,nlwc->nlw", feat_s1, feat_s1_unfold) / feat_s1_unfold.shape[2]

        feat_s0_fea = torch.einsum("nlwc,nlw->nlc", feat_s0_unfold, feat_s0_sem)
        feat_s1_fea = torch.einsum("nlwc,nlw->nlc", feat_s1_unfold, feat_s1_sem)

        norm_0 = conf_mask.sum(dim=2, keepdim=True).clamp(min=1e-6)
        feat_s0_sem_fea = torch.einsum("nls,nsc->nlc", conf_mask, feat_s1_fea) / norm_0

        conf_mask_trans = torch.transpose(conf_mask, -2, -1)
        norm_1 = conf_mask_trans.sum(dim=2, keepdim=True).clamp(min=1e-6)
        feat_s1_sem_fea = torch.einsum("nsl,nlc->nsc", conf_mask_trans, feat_s0_fea) / norm_1

        # 【【【 修复点 】】】
        # "nlc,nsc->nlc" (错误) 更改为 "nlc,nsc->nls" (正确)
        # 这将创建 [N, L, S] 形状的矩阵 (即 [2, 1600, 1600])
        conf_matrix_fusion = torch.einsum("nlc,nsc->nls", feat_s0_sem_fea,
                                          feat_s1_sem_fea) / self.d_model ** .5
        data['conf_matrix_fusion'] = conf_matrix_fusion

        # print(conf_matrix_fusion.shape) # 现在应该打印 [2, 1600, 1600]

        # 这里的 AdaptiveAvgPool1d 作用于最后一个维度 (C)
        # [N, L, C] -> [N, L, 160]
        feat_s0_pooled = self.avgpool(feat_s0)
        feat_s0_sem_fea_pooled = self.avgpool_1(feat_s0_sem_fea)
        # [N, S, C] -> [N, S, 160]
        feat_s1_pooled = self.avgpool(feat_s1)
        feat_s1_sem_fea_pooled = self.avgpool_1(feat_s1_sem_fea)

        # [N, L, 160] + [N, L, 96] -> [N, L, 256]
        feat_s0_out = torch.concat((feat_s0_pooled, feat_s0_sem_fea_pooled), -1)
        feat_s1_out = torch.concat((feat_s1_pooled, feat_s1_sem_fea_pooled), -1)

        feat_s = torch.concat((feat_s0_out, feat_s1_out), 0)
        # LayerNorm(256) 作用于 [N+N, L/S, 256], 这是正确的
        feat_s = self.norm_feat(feat_s)

        return feat_s[:feat_s0_out.shape[0]], feat_s[feat_s0_out.shape[0]:]


import copy
import torch
import torch.nn as nn
# 假设 LoFTREncoderLayer 在其他地方定义 (虽然在此模块中不再需要)
# from .linear_attention import LinearAttention, FullAttention
import math
from einops import rearrange
import torch.nn.functional as F


class Relator_Fusion(nn.Module):
    """
    A module to fuse multiple similarity matrices (relators) and refine
    the input features based on this fused similarity context.

    The original LoFTR module description might be a placeholder/misnomer.
    """

    def __init__(self, config):
        super(Relator_Fusion, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.d_model_relator = config['d_model']
        self.up_dim_ini = config['relator_dim_ini']

        # Pooling layers to standardize feature lengths
        self.avgpool = nn.AdaptiveAvgPool1d(160)  # For input features
        self.avgpool_1 = nn.AdaptiveAvgPool1d(96)  # For similarity features

        # Removed commented-out "dead code" (down_dim*, layers)

        self.norm_feat = nn.LayerNorm(self.d_model)  # Final LayerNorm
        self.head = 4
        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'temp' in name or 'sample_offset' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C] (e.g., from FeatureFusion)
            feat1 (torch.Tensor): [N, S, C] (e.g., from FeatureFusion)
            data (dict): Dictionary containing pre-computed similarity matrices
                         {'conf_matrix': [N, L, S], 'conf_matrix_fusion': [N, L, S]}
        Outputs:
            feat0: [N, L, C] (Refined feature)
            feat1: [N, S, C] (Refined feature)
        """

        # --- 1. Fuse Similarity Matrices ---

        # [OPTIMIZATION] Removed redundant calculation of 'relator'
        # Assumes data['conf_matrix'] is the same as torch.einsum("nlc,nsc->nls", feat0, feat1)
        # relator = torch.einsum("nlc,nsc->nls", feat0, feat1) / self.d_model ** .5
        # relator = torch.unsqueeze(relator, 1)

        # Get pre-computed similarity matrices from data dict
        conf_matrix_fusion = torch.unsqueeze(data['conf_matrix_fusion'], 1)  # [N, 1, L, S]
        conf_matrix = torch.unsqueeze(data['conf_matrix'], 1)  # [N, 1, L, S]

        # [OPTIMIZATION] Use conf_matrix in place of the redundant 'relator'
        # Concatenate the three similarity views
        relators = torch.concat((conf_matrix, conf_matrix, conf_matrix_fusion), 1)  # [N, 3, L, S]

        # (Optional) Store the mean similarity for external use (e.g., loss)
        data["conf_matrix_relator"] = torch.squeeze(torch.mean(relators, 1), 1)

        # --- 2. Create "Similarity Context" Feature ---

        # Symmetrize by concatenating A->B and B->A similarities
        relator_trans = torch.transpose(relators, -1, -2)  # [N, 3, S, L]
        # 【修复】L==S 假设: 仅在 L==S 时才 concat
        N, L, S = conf_matrix.shape[0], conf_matrix.shape[2], conf_matrix.shape[3]
        if L == S:
            relators = torch.concat((relators, relator_trans), 0)  # [2N, 3, L, L]
            feat = torch.concat((feat0, feat1), 0)  # [2N, L, C]
        else:
            # 如果 L != S, 我们不能简单地 concat,
            # 我们需要单独处理 feat0 和 feat1
            # (这是一个简化的处理，更复杂的逻辑可能需要 padding/unpadding)
            print(f"Warning (Relator_Fusion): L ({L}) != S ({S}). Using asymmetric fusion.")
            # --- 处理 feat0 (使用 A->B 相似度) ---
            relators0 = torch.reshape(relators, (N, L, S * 3))  # [N, L, S*3]
            relators0 = self.avgpool_1(relators0)  # [N, L, 96]
            feat0_pooled = self.avgpool(feat0)  # [N, L, 160]
            feat0_out = torch.concat((feat0_pooled, relators0), -1)
            feat0_out = self.norm_feat(feat0_out)

            # --- 处理 feat1 (使用 B->A 相似度) ---
            relators1 = torch.reshape(relator_trans, (N, S, L * 3))  # [N, S, L*3]
            relators1 = self.avgpool_1(relators1)  # [N, S, 96]
            feat1_pooled = self.avgpool(feat1)  # [N, S, 160]
            feat1_out = torch.concat((feat1_pooled, relators1), -1)
            feat1_out = self.norm_feat(feat1_out)

            return feat0_out, feat1_out
            # --- L!=S 分支结束 ---

        # --- 3. (L==S 分支) ---
        # Reshape to [2N, L, 3*L]
        relators = torch.reshape(relators,
                                 (relators.shape[0], relators.shape[2], relators.shape[3] * relators.shape[1]))

        # Pool to fixed dimension [2N, L, 96]
        relators = self.avgpool_1(relators)

        # --- 3. Process Input Features ---
        # (feat [2N, L, C] 已在 L==S 分支中 concat)
        # Pool features to fixed dimension [2N, L, 160]
        feat = self.avgpool(feat)

        # --- 4. Final Fusion and Refinement ---

        # Concatenate features with similarity context [2N, L, 160+96]
        feat = torch.concat((feat, relators), -1)

        # Apply LayerNorm (Assumes d_model == 160+96 == 256)
        feat = self.norm_feat(feat)

        # Unstack the batch
        feat0 = feat[:feat0.shape[0]]  # [N, L, C]
        feat1 = feat[feat0.shape[0]:]  # [N, S, C]

        return feat0, feat1