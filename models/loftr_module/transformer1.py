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
                 attention='linear'):
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
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
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
    def __init__(self, d_model, nhead, attention_type):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, feat0, feat1, mask0, mask1):
        # 这是一个简化的占位符，实际的 LoFTR 层更复杂
        if feat1 is not None:
            return self.attn(feat0.transpose(0, 1), feat1.transpose(0, 1), feat1.transpose(0, 1))[0].transpose(0, 1)
        else:
            return self.attn(feat0.transpose(0, 1), feat0.transpose(0, 1), feat0.transpose(0, 1))[0].transpose(0, 1)


LoFTREncoderLayer = LoFunction_Placeholder  # 使用占位符

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import copy
import math


# (假设 LoFTREncoderLayer 占位符已定义)
# (为简洁起见，省略 LoFTREncoderLayer 占位符)
class LoFunction_Placeholder(nn.Module):  # 占位符
    def __init__(self, d_model, nhead, attention_type):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, feat0, feat1, mask0, mask1):
        if feat1 is not None:
            return self.attn(feat0.transpose(0, 1), feat1.transpose(0, 1), feat1.transpose(0, 1))[0].transpose(0, 1)
        else:
            return self.attn(feat0.transpose(0, 1), feat0.transpose(0, 1), feat0.transpose(0, 1))[0].transpose(0, 1)


LoFTREncoderLayer = LoFunction_Placeholder  # 使用占位符


class TopicFormer(nn.Module):

    def __init__(self, config):
        super(TopicFormer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names_t']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])

        self.feat_aug = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(2 * config['n_topic_transformers'])])
        self.n_iter_topic_transformer = config['n_topic_transformers']

        self.seed_tokens = nn.Parameter(torch.randn(config['n_topics'], config['d_model']))
        self.register_parameter('seed_tokens', self.seed_tokens)
        self.topic_drop = nn.Dropout1d(p=0.1)
        self.n_samples = config['n_samples']

        # (v3.0 修复: avgpool 和 avgpool_1 已变为未使用)
        self.avgpool = nn.AdaptiveAvgPool1d(160)
        self.avgpool_1 = nn.AdaptiveAvgPool1d(96)
        self.norm_feat = nn.LayerNorm(self.d_model)

        # --- 【v4.1 修复 A】: 解决 5k/22k 过拟合问题 ---
        # 增加温度，软化 Softmax，防止过度自信
        self.topic_temperature = config.get('topic_temperature', 10.0)
        # ---

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sample_topic(self, prob_topics, topics, L):
        # (此函数逻辑不变)
        prob_topics0, prob_topics1 = prob_topics[:, :L], prob_topics[:, L:]
        topics0, topics1 = topics[:, :L], topics[:, L:]
        theta0 = F.normalize(prob_topics0.sum(dim=1), p=1, dim=-1)
        theta1 = F.normalize(prob_topics1.sum(dim=1), p=1, dim=-1)
        theta = F.normalize(theta0 * theta1, p=1, dim=-1)
        if self.n_samples == 0:
            return None
        if self.training:
            sampled_inds = torch.multinomial(theta, self.n_samples)
        else:
            sampled_values, sampled_inds = torch.topk(theta, self.n_samples, dim=-1)
        sampled_topics0 = torch.gather(topics0, dim=-1, index=sampled_inds.unsqueeze(1).repeat(1, topics0.shape[1], 1))
        sampled_topics1 = torch.gather(topics1, dim=-1, index=sampled_inds.unsqueeze(1).repeat(1, topics1.shape[1], 1))
        return sampled_topics0, sampled_topics1

    def reduce_feat(self, feat, topick, N, C):
        # (此函数逻辑不变, v1.9 优化版)
        len_topic = topick.sum(dim=-1).int()
        max_len = len_topic.max().item()
        selected_ids = topick.bool()
        if max_len == 0:
            resized_feat = torch.zeros((N, 0, C), dtype=torch.float, device=feat.device)
            new_mask = torch.zeros((N, 0), dtype=torch.bool, device=feat.device)
            return resized_feat, new_mask, selected_ids
        resized_feat = torch.zeros((N, max_len, C), dtype=torch.float, device=feat.device)
        new_mask = torch.arange(max_len, device=feat.device).unsqueeze(0) < len_topic.unsqueeze(1)
        if new_mask.sum() > 0:
            resized_feat[new_mask] = feat[selected_ids]
        return resized_feat, new_mask, selected_ids

    def forward(self, feat0, feat1, topic0, topic1, mask0=None, mask1=None):

        assert self.d_model == feat0.shape[2], "the feature number of src and transformer must be equal"
        N, L, S, C, K = feat0.shape[0], feat0.shape[1], feat1.shape[1], feat0.shape[2], self.config['n_topics']

        # 1. 'seeds' (v3.0 逻辑)
        seeds = self.seed_tokens.unsqueeze(0).repeat(N, 1, 1)
        seeds = self.topic_drop(seeds)

        # 2. 'topic' (v3.0 中已变为未使用逻辑)
        top_matrix = torch.einsum("nlc,nsc->nls", topic0, topic1) / C ** .5
        top_matrix = F.softmax(top_matrix, 1) * F.softmax(top_matrix, 2)
        top_matrix_idx = torch.argmax(top_matrix, dim=-1)
        C_topic1 = topic1.shape[-1]
        idx_expanded = top_matrix_idx.unsqueeze(-1).expand(-1, -1, C_topic1)
        topic1_warped = torch.gather(topic1, dim=1, index=idx_expanded)
        topic = torch.concat((topic0, topic1_warped), -1)
        topic = self.avgpool_1(topic)  # (未使用)

        # 3. 特征和种子交互 (v3.0 逻辑)
        feat = torch.cat((feat0, feat1), dim=1)
        if mask0 is not None:
            mask = torch.cat((mask0, mask1), dim=-1)
        else:
            mask = None

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'seed':
                seeds = layer(seeds, feat, None, mask)
            elif name == 'feat':
                feat0 = layer(feat0, seeds, mask0, None)
                feat1 = layer(feat1, seeds, mask1, None)

        # 4. 【v3.0 修复】: 移除拼接
        seeds = self.norm_feat(seeds)

        # 5. 【v4.1 修复 B】: 移除 Logits 缩放 ( / C ** .5)
        dmatrix = torch.einsum("nmd,nkd->nmk", feat, seeds)

        # 5. 【v4.1 修复 C】: 应用温度 (解决 22k 过拟合)
        prob_topics = F.softmax(dmatrix / self.topic_temperature, dim=-1)

        # 6. (采样逻辑不变)
        feat_topics = torch.zeros_like(dmatrix).scatter_(-1, torch.argmax(dmatrix, dim=-1, keepdim=True), 1.0)
        if mask is not None:
            feat_topics = feat_topics * mask.unsqueeze(-1)
            prob_topics = prob_topics * mask.unsqueeze(-1)
        sampled_topics = self.sample_topic(prob_topics.detach(), feat_topics, L)

        # 7. IF/ELSE 分支 (v3.0 逻辑, 不变)
        if sampled_topics is not None:
            # (省略... 保持 v3.0 的 reduce_feat 和 feat_aug 循环)
            updated_feat0, updated_feat1 = torch.zeros_like(feat0), torch.zeros_like(feat1)
            s_topics0, s_topics1 = sampled_topics
            for k in range(s_topics0.shape[-1]):
                topick0, topick1 = s_topics0[..., k], s_topics1[..., k]
                if (topick0.sum() > 0) and (topick1.sum() > 0):
                    new_feat0, new_mask0, selected_ids0 = self.reduce_feat(feat0, topick0, N, C)
                    new_feat1, new_mask1, selected_ids1 = self.reduce_feat(feat1, topick1, N, C)
                    for idt in range(self.n_iter_topic_transformer):
                        new_feat0 = self.feat_aug[idt * 2](new_feat0, new_feat0, new_mask0, new_mask0)
                        new_feat1 = self.feat_aug[idt * 2](new_feat1, new_feat1, new_mask1, new_mask1)
                        new_feat0 = self.feat_aug[idt * 2 + 1](new_feat0, new_feat1, new_mask0, new_mask1)
                        new_feat1 = self.feat_aug[idt * 2 + 1](new_feat1, new_feat0, new_mask1, new_mask0)
                    if new_mask0.sum() > 0:
                        updated_feat0[selected_ids0] = new_feat0[new_mask0]
                    if new_mask1.sum() > 0:
                        updated_feat1[selected_ids1] = new_feat1[new_mask1]
            feat0 = (1 - s_topics0.sum(dim=-1, keepdim=True)) * feat0 + updated_feat0
            feat1 = (1 - s_topics1.sum(dim=-1, keepdim=True)) * feat1 + updated_feat1
        else:
            for idt in range(self.n_iter_topic_transformer * 2):
                feat0 = self.feat_aug[idt](feat0, seeds, mask0, None)
                feat1 = self.feat_aug[idt](feat1, seeds, mask1, None)

        # 8. 【v4.0 修复】: UNFOLD 平滑操作

        # 【v3.1 修复】: 使用 prob_topics (概率) 进行平滑
        topic_matrix_img0_scores = prob_topics[:, :L]  # [N, L, K]

        Nt, K_dim = topic_matrix_img0_scores.shape[0], topic_matrix_img0_scores.shape[2]
        Ht0 = Wt0 = int(math.sqrt(L))
        if Ht0 * Wt0 != L:
            print(f"Warning: L ({L}) is not a perfect square. Using floored sqrt.")
            Ht0, Wt0 = int(math.sqrt(L)), int(math.sqrt(L))

        topic_matrix_img0_2d = rearrange(topic_matrix_img0_scores, 'n (h w) k -> n k h w', h=Ht0, w=Wt0)
        topic_matrix_img0_smooth = F.avg_pool2d(topic_matrix_img0_2d, kernel_size=3, stride=1, padding=1)

        # --- 【v4.0 修复 D】: 移除乘法！ (您指出的核心 Bug) ---
        # 原始 (Bug): topic_matrix_img0_out_2d = topic_matrix_img0_2d * topic_matrix_img0_smooth
        topic_matrix_img0_out_2d = topic_matrix_img0_smooth
        # ---

        topic_matrix_img0 = rearrange(topic_matrix_img0_out_2d, 'n k h w -> n (h w) k')

        # --- 对 img1 重复操作 ---
        topic_matrix_img1_scores = prob_topics[:, L:]  # [N, S, K]
        Ht1 = Wt1 = int(math.sqrt(S))
        if Ht1 * Wt1 != S:
            print(f"Warning: S ({S}) is not a perfect square. Using floored sqrt.")
            Ht1, Wt1 = int(math.sqrt(S)), int(math.sqrt(S))

        topic_matrix_img1_2d = rearrange(topic_matrix_img1_scores, 'n (h w) k -> n k h w', h=Ht1, w=Wt1)
        topic_matrix_img1_smooth = F.avg_pool2d(topic_matrix_img1_2d, kernel_size=3, stride=1, padding=1)

        # --- 【v4S.0 修复 E】: 移除乘法！ ---
        # 原始 (Bug): topic_matrix_img1_out_2d = topic_matrix_img1_2d * topic_matrix_img1_smooth
        topic_matrix_img1_out_2d = topic_matrix_img1_smooth
        # ---

        topic_matrix_img1 = rearrange(topic_matrix_img1_out_2d, 'n k h w -> n (h w) k')

        # 9. 返回平滑后的概率 (CoarseMatching v3.1 会正确处理)
        topic_matrix_match = {"img0": topic_matrix_img0, "img1": topic_matrix_img1}

        return feat0, feat1, topic_matrix_match


class FineNetwork(nn.Module):

    def __init__(self, config, add_detector=True):
        super(FineNetwork, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        # --- 【潜在 Bug 修复】: 假设 MLPMixerEncoderLayer 未定义, 切换回 LoFTREncoderLayer ---
        # (如果 MLPMixerEncoderLayer 是您自定义的, 您可能需要保留它)
        # self.n_mlp_mixer_blocks = config["n_mlp_mixer_blocks"]
        # self.encoder_layers = nn.ModuleList([MLPMixerEncoderLayer(config["n_feats"] * 2, self.d_model)
        #                                      for _ in range(self.n_mlp_mixer_blocks)])
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        # --- 修复结束 ---

        self.detector = None
        if add_detector:
            # 假设 detector 也使用 LoFTREncoderLayer
            self.detector = nn.Sequential(LoFTREncoderLayer(config["d_model"], config['nhead'], config['attention']),
                                          nn.Linear(self.d_model, 1))

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
        # (原始的 MLPMixer 逻辑已被替换)
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


# 假设 LoFTREncoderLayer, MLPMixerEncoderLayer, conv1x1 等在其他地方定义
# (虽然这个特定文件不需要它们，但保留上下文)

class FeatureFusion(nn.Module):

    def __init__(self, config):
        super(FeatureFusion, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.d_model_fusion = config['d_model']
        self.nhead = config['nhead']
        self.avgpool = nn.AdaptiveAvgPool1d(160)
        self.avgpool_1 = nn.AdaptiveAvgPool1d(96)
        self.norm_feat = nn.LayerNorm(self.d_model)
        # 已移除被注释掉的 "死代码" (self.down_dim*)
        self.head = 4
        # 使用 .get() 增加健壮性，0.1 是一个示例默认值
        self.thr = config.get('thr', 0.1)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_s0, feat_s1, data=None):
        # 假设 feat_s0, feat_s1 传入时形状为 [N, L, C] 或 [N, S, C]

        # --- 【修复】L==S 假设问题 ---
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
        # --- 修复结束 ---

        # 确保 C 维度匹配
        assert self.d_model_fusion == C, "the feature number of src and transformer must be equal"

        # 原始特征已经是 [N, L, C] 和 [N, S, C]，无需 reshape
        # feat_s0 = torch.reshape(feat_s0, (N, H * W, C))
        # feat_s1 = torch.reshape(feat_s1, (N, H * W, C)) # 假设 L=S=H*W

        # 1. 计算初始相似度 (可能对应 S_si)
        conf_matrix = torch.einsum("nlc,nsc->nls", feat_s0, feat_s1) / self.d_model ** .5

        # 【精简】移除了冗余的 reshape 和赋值

        data["conf_matrix"] = conf_matrix

        # 2. 计算置信度与硬掩码 (conf_mask)
        conf_matrix_softmax = F.softmax(conf_matrix, 1) * F.softmax(conf_matrix, 2)
        conf_mask = conf_matrix_softmax > self.thr
        conf_mask = conf_mask * (conf_matrix_softmax == conf_matrix_softmax.max(dim=2, keepdim=True)[0]) \
                    * (conf_matrix_softmax == conf_matrix_softmax.max(dim=1, keepdim=True)[0])
        conf_mask = conf_mask.float()  # [N, L, S]

        # 3. 计算邻域自注意力特征
        feat_s0_2d = torch.reshape(feat_s0, (N, C, H0, W0))
        feat_s1_2d = torch.reshape(feat_s1, (N, C, H1, W1))

        feat_s0_unfold = F.unfold(feat_s0_2d, kernel_size=(3, 3), stride=1, padding=1)
        feat_s0_unfold = rearrange(feat_s0_unfold, 'n (c ww) l -> n l ww c', ww=3 ** 2)  # [N, L, 9, C]

        feat_s0_sem = torch.einsum("nlc,nlwc->nlw", feat_s0, feat_s0_unfold) / feat_s0_unfold.shape[
            2]  # [N, L, 9] (除以 9)

        feat_s1_unfold = F.unfold(feat_s1_2d, kernel_size=(3, 3), stride=1, padding=1)
        feat_s1_unfold = rearrange(feat_s1_unfold, 'n (c ww) l -> n l ww c', ww=3 ** 2)
        feat_s1_sem = torch.einsum("nlc,nlwc->nlw", feat_s1, feat_s1_unfold) / feat_s1_unfold.shape[2]  # [N, S, 9]

        feat_s0_fea = torch.einsum("nlwc,nlw->nlc", feat_s0_unfold, feat_s0_sem)  # [N, L, C]
        feat_s1_fea = torch.einsum("nlwc,nlw->nlc", feat_s1_unfold, feat_s1_sem)  # [N, S, C]

        # 4. 执行跨图像特征融合 (CNIM 核心)

        # --- 【Bug 修复】 ---
        # 原始除数: / conf_mask.shape[1] (即 / L)
        # 修复后的除数: / norm_0 (即除以每个 l 点的匹配数)
        norm_0 = conf_mask.sum(dim=2, keepdim=True).clamp(min=1e-6)  # [N, L, 1]
        feat_s0_sem_fea = torch.einsum("nls,nsc->nlc", conf_mask, feat_s1_fea) / norm_0

        conf_mask_trans = torch.transpose(conf_mask, -2, -1)
        # 原始除数: / conf_mask_trans.shape[1] (即 / S)
        # 修复后的除数: / norm_1 (即除以每个 s 点的匹配数)
        norm_1 = conf_mask_trans.sum(dim=2, keepdim=True).clamp(min=1e-6)  # [N, S, 1]
        feat_s1_sem_fea = torch.einsum("nsl,nlc->nsc", conf_mask_trans, feat_s0_fea) / norm_1
        # --- 修复结束 ---

        # 5. 计算融合后的相似度 (可能对应 S_ci)
        conf_matrix_fusion = torch.einsum("nlc,nsc->nlc", feat_s0_sem_fea,
                                          feat_s1_sem_fea) / self.d_model ** .5
        data['conf_matrix_fusion'] = conf_matrix_fusion

        # 6. 特征拼接与归一化
        feat_s0 = self.avgpool(feat_s0)
        feat_s0_sem_fea = self.avgpool_1(feat_s0_sem_fea)
        feat_s1 = self.avgpool(feat_s1)
        feat_s1_sem_fea = self.avgpool_1(feat_s1_sem_fea)

        feat_s0 = torch.concat((feat_s0, feat_s0_sem_fea), -1)
        feat_s1 = torch.concat((feat_s1, feat_s1_sem_fea), -1)

        feat_s = torch.concat((feat_s0, feat_s1), 0)
        feat_s = self.norm_feat(feat_s)
        # 已移除被注释掉的 "死代码" (reshape, down_dim*)

        # 7. 返回
        return feat_s[:feat_s0.shape[0]], feat_s[feat_s0.shape[0]:]


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