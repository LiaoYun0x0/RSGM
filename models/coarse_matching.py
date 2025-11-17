import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat
from kornia.utils import create_meshgrid
from numpy import dtype

from configs.default import get_cfg_defaults, lower_config

INF = 1e9
config = get_cfg_defaults()
config = lower_config(config)


def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseMatching2(nn.Module):
    """
    【推荐架构】: 视觉为主，主题为辅 (Soft Fusion)

    这个版本解决了 Gumbel-Topic-Matcher 的收敛问题。
    它不使用“硬分配”门控 (Hard Gating)，而是：
    1. 计算视觉相似度 (sim_visual) 作为基础。
    2. 计算“软”主题相似度 (topic_sim) 作为辅助。
    3. 将两者相加融合: logits_final = sim_visual + (w * topic_sim)
    4. 在融合后的稠密矩阵上执行 MNN。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eps = 1e-8

        # Gumbel-Softmax 或 Softmax 的温度。
        # 用于 "topic_sim" 的计算。
        self.tau = config.get('gumbel_tau', 1.0)

        # 【核心】可学习的融合权重 (w)
        # 我们将其初始化为 0.0。
        # 这意味着在训练刚开始时，w=0，
        # logits_final = sim_visual + 0 * topic_sim = sim_visual
        # 此时，模型 *等同于* 你那个工作良好的版本1 (纯视觉)。
        # 然后，优化器会 *只在* topic_sim 有助于降低损失的情况下，
        # 才“学会”给 self.topic_weight 一个非零值。
        # 这保证了训练的绝对稳定。
        self.topic_weight = nn.Parameter(torch.tensor(0.0))

    def compute_similarity_matrix(self, feat_q, feat_r):
        """辅助函数：只计算视觉相似度"""
        return torch.matmul(feat_q, feat_r.transpose(1, 2))  # (N, L, S)

    def forward(self, feat_c0_raw, feat_c1_raw, feat_c0, feat_c1,
                topic_matrix_match, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat_c0, feat_c1: 视觉特征 (来自 TopicFormer)
            topic_matrix_match (dict): 包含 *Logits* 的字典
                                     {'img0': [N,L,K], 'img1': [N,S,K]}
            data (dict): 要更新的数据字典
        """

        # 1. 【主路径】计算基础视觉相似度 (使用增强后的特征)
        sim_visual = self.compute_similarity_matrix(feat_c0, feat_c1)  # [N, L, S]

        # 2. 【辅助路径】计算“软”主题相似度
        topic_logits_0 = topic_matrix_match['img0']
        topic_logits_1 = topic_matrix_match['img1']

        # 使用 F.softmax (Gumbel-Softmax的 "soft" 形式)
        # tau 控制了分布的“尖锐”程度
        soft_assign_0 = F.softmax(topic_logits_0 / self.tau, dim=-1)  # [N, L, K]
        soft_assign_1 = F.softmax(topic_logits_1 / self.tau, dim=-1)  # [N, S, K]

        # 'nlk, nsk -> nls'
        # topic_sim[n, l, s] 是一个 [0, 1] 之间的值，
        # 表示点 'l' 和点 's' 在主题分布上的相似程度。
        topic_sim = torch.einsum('nlk, nsk -> nls', soft_assign_0, soft_assign_1)  # [N, L, S]

        # 3. 【核心】融合
        # 视觉为主 (sim_visual)，主题为辅 (self.topic_weight * topic_sim)
        # self.topic_weight 会从 0 开始学习，自动调整辅助路径的强度
        logits_final = sim_visual + self.topic_weight * topic_sim

        # --- MNN 匹配 (逻辑与版本1相同) ---
        # 关键：MNN 现在作用于一个“稠密”的、融合了主题信息的矩阵上，
        # 而不是那个被硬分配“破坏”掉的稀疏矩阵。
        conf = torch.softmax(logits_final, 1) * torch.softmax(logits_final, 2)

        # 4. 数值稳定归一化 (从版本1保留)
        max_val = conf.max().detach() * 1.2
        conf = conf / (max_val + self.eps)  # 增加稳定性

        # 5. MNN 过滤
        conf_f = conf * (
                (conf == conf.max(dim=2, keepdim=True)[0]) &
                (conf == conf.max(dim=1, keepdim=True)[0])
        )

        # 6. 匹配点提取
        mask_v, all_j_ids = conf_f.max(dim=2)
        b_ids, i_ids = torch.where(mask_v > 0)  # 确保有置信度
        j_ids = all_j_ids[b_ids, i_ids]
        matches = torch.stack([b_ids, i_ids, j_ids]).T

        # 7. 更新 data 字典
        # 【重要修复】：这里存储 'conf' (概率)，而不是 'logits_final'。
        # 这与你版本1的行为一致，并确保下游损失函数拿到的是正确的输入。
        data.update({
            'conf_matrix': conf,
            'matches': matches
        })
        return data

class CoarseMatching1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def compute_confidence_matrix(self, query_lf, refer_lf, cf_matrix=None):
        similarity_matrix = torch.matmul(query_lf, refer_lf.transpose(1, 2))
        confidence_matrix = torch.softmax(similarity_matrix, 1) * torch.softmax(similarity_matrix, 2)
        confidence_matrix = confidence_matrix / (math.fabs(torch.max(confidence_matrix).item()) * 1.2)
        confidence_matrix_f = (confidence_matrix * (
                confidence_matrix == confidence_matrix.max(dim=2, keepdim=True)[0]) * (
                                       confidence_matrix == confidence_matrix.max(dim=1, keepdim=True)[0]))
        return confidence_matrix, confidence_matrix_f

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)
        H = int(math.sqrt(L))

        conf_matrix_s, conf_matrix_s_f = self.compute_confidence_matrix(feat_c0, feat_c1)  # (C * temperature)
        mask_v, all_j_ids = conf_matrix_s_f.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        matches = torch.stack([b_ids, i_ids, j_ids]).T

        data.update({'conf_matrix': conf_matrix_s})
        data.update({'matches': matches})


import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseMatching(nn.Module):
    """
    【推荐架构】: 温度退火 + 软融合

    这个版本实现了“随着训练深入，主题辅助作用越来越明显”的需求。

    它通过“温度退火” (Temperature Annealing) 来实现：
    1. 训练初期, tau 很高, 主题分配很“软” (模糊), 辅助作用很弱。
    2. 随着 global_step 增加, tau 逐渐降低 (退火)。
    3. 训练后期, tau 很低, 主题分配很“硬” (自信), 辅助作用变强。

    同时，它保留了从0开始学习的 self.topic_weight，以确保训练初期的绝对稳定。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eps = 1e-8

        # --- 【核心修改】温度退火参数 ---
        # 1. 从 config 中读取退火的超参数
        # 初始温度 (高, 产生“软”分配)
        self.tau_start = config.get('gumbel_tau_start', 1.0)
        # 最终温度 (低, 产生“硬”分配)
        self.tau_end = config.get('gumbel_tau_end', 0.1)
        # 达到最终温度所需的训练步数
        self.tau_anneal_steps = config.get('gumbel_anneal_steps', 10000)
        # ---

        # 【保留】可学习的融合权重
        # 仍然强烈建议从 0.0 开始，以确保模型在 *学会* # 如何使用主题信息之前，完全依赖于稳定的视觉匹配 (版本1)。
        self.topic_weight = nn.Parameter(torch.tensor(0.0))

    def compute_similarity_matrix(self, feat_q, feat_r):
        """辅助函数：只计算视觉相似度"""
        return torch.matmul(feat_q, feat_r.transpose(1, 2))  # (N, L, S)

    def forward(self, feat_c0_raw, feat_c1_raw, feat_c0, feat_c1,
                topic_matrix_match, data, mask_c0=None, mask_c1=None,
                # 【重要】: 必须从你的训练循环中传入 global_step
                global_step=None):
        """
        Args:
            ...
            global_step (int, optional): 当前的全局训练步数。
        """

        # 1. 【核心】计算当前的退火温度 (current_tau)
        current_tau = self.tau_end
        if self.training and global_step is not None:
            # 线性退火 (Linear Annealing)
            progress = min(1.0, global_step / self.tau_anneal_steps)
            current_tau = self.tau_start + progress * (self.tau_end - self.tau_start)
        elif self.training:
            # 如果没传入 global_step，则使用起始 tau
            current_tau = self.tau_start
        # (在
        #  eval 模式下，自动使用 self.tau_end)

        # 2. 【主路径】计算基础视觉相似度
        sim_visual = self.compute_similarity_matrix(feat_c0, feat_c1)  # [N, L, S]

        # 3. 【辅助路径】使用 *退火后的 current_tau* 计算主题相似度
        topic_logits_0 = topic_matrix_match['img0']
        topic_logits_1 = topic_matrix_match['img1']

        # 使用 current_tau，随着训练，这里的输出会越来越“硬”
        soft_assign_0 = F.softmax(topic_logits_0 / current_tau, dim=-1)  # [N, L, K]
        soft_assign_1 = F.softmax(topic_logits_1 / current_tau, dim=-1)  # [N, S, K]

        # 主题分布越“硬”，topic_sim 矩阵的对比度就越强
        topic_sim = torch.einsum('nlk, nsk -> nls', soft_assign_0, soft_assign_1)  # [N, L, S]

        # 4. 【核心】融合
        # self.topic_weight 保证了模型 *可以* 学习辅助作用
        # current_tau 保证了该作用随着训练 *自动* 增强
        logits_final = sim_visual + self.topic_weight * topic_sim

        # --- MNN 匹配 (逻辑与版本1相同) ---
        conf = torch.softmax(logits_final, 1) * torch.softmax(logits_final, 2)

        # 5. 数值稳定归一化
        max_val = conf.max().detach() * 1.2
        conf = conf / (max_val + self.eps)

        # 6. MNN 过滤
        conf_f = conf * (
                (conf == conf.max(dim=2, keepdim=True)[0]) &
                (conf == conf.max(dim=1, keepdim=True)[0])
        )

        # 7. 匹配点提取
        mask_v, all_j_ids = conf_f.max(dim=2)
        b_ids, i_ids = torch.where(mask_v > 0)
        j_ids = all_j_ids[b_ids, i_ids]
        matches = torch.stack([b_ids, i_ids, j_ids]).T

        # 8. 更新 data 字典
        data.update({
            'conf_matrix': conf,
            'matches': matches
            # 'current_tau': current_tau # (可选) 方便调试
        })
        return data