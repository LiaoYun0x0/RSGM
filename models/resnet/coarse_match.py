import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import copy
import math

INF = 1e9


# --- Min-Max 归一化函数 ---
def normalize_min_max(matrix, dim=(1, 2), eps=1e-9):
    """
    对输入的 (N, L, S) 或 (N, K, L, S) 矩阵在指定的
    维度上进行 Min-Max 归一化，将其缩放到 [0, 1] 范围。
    """
    if matrix.ndim <= max(dim):
        # print(f"Warning: Matrix ndim ({matrix.ndim}) is too small for normalization dims {dim}. Skipping.")
        return matrix
    try:
        min_val = torch.amin(matrix, dim=dim, keepdim=True)
        max_val = torch.amax(matrix, dim=dim, keepdim=True)
        matrix_range = max_val - min_val
        # 避免除以零
        matrix_range[matrix_range < eps] = eps
        normalized_matrix = (matrix - min_val) / matrix_range
        # 裁剪到 [0, 1] 范围
        normalized_matrix = torch.clamp(normalized_matrix, 0, 1)
    except Exception as e:
        # print(f"Error during normalization: {e}. Returning original matrix.")
        normalized_matrix = matrix
    return normalized_matrix


# --- Sinkhorn 占位符 ---
try:
    # 假设 log_optimal_transport 是从其他地方导入的
    # from ... import log_optimal_transport

    # 作为示例的占位符:
    def log_optimal_transport(scores, bin_score, iters):
        # print("Warning: Using placeholder log_optimal_transport!")
        N, L, S = scores.shape
        # 返回一个模拟的 (N, L+1, S+1) 输出
        return torch.rand(N, L + 1, S + 1, device=scores.device)
except ImportError:
    log_optimal_transport = None


class CoarseMatching(nn.Module):
    """
    重构后的粗匹配模块。

    该模块不再自己寻找匹配，而是计算一个精炼的
    "置信度矩阵" (conf_matrix)，供 model.py 后续使用。

    它融合了:
    1. S_feat: 精炼特征 (来自 TopicFormer) 间的相似度。
    2. S_topic: 主题概率 (来自 TopicFormer) 间的相似度。

    并使用残差门控 (Residual Gating) 将它们组合。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.thr = config.get('thr', 0.2)
        self.border_rm = config.get('border_rm', 0)

        self.match_type = config.get('match_type', 'dual_softmax')

        if self.match_type == 'dual_softmax':
            self.temperature = config.get('dsmax_temperature', 0.1)
        elif self.match_type == 'sinkhorn':
            if log_optimal_transport is None:
                raise ImportError("Sinkhorn matching requires log_optimal_transport function.")
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(
                torch.tensor(config.get('skh_init_bin_score', 1.0), requires_grad=True))
            self.skh_iters = config.get('skh_iters', 3)
            self.skh_prefilter = config.get('skh_prefilter', False)
        else:
            raise NotImplementedError(f"Match type {self.match_type} not implemented.")

    def forward(self, feat_c0, feat_c1, feat_c00, feat_c10, topic_matrix, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat_c0 (torch.Tensor): [N, L, C] 原始粗特征 (带位置编码)
            feat_c1 (torch.Tensor): [N, S, C] 原始粗特征 (带位置编码)
            feat_c00 (torch.Tensor): [N, L, C] TopicFormer 精炼后的特征
            feat_c10 (torch.Tensor): [N, S, C] TopicFormer 精炼后的特征
            topic_matrix (dict): 包含 'img0': [N, L, K] 和 'img1': [N, S, K] 的概率
            data (dict): 用于存储输出 'conf_matrix' 的字典
            mask_c0 (torch.Tensor, optional): [N, L]
            mask_c1 (torch.Tensor, optional): [N, S]
        """
        N, L, S, C = feat_c00.size(0), feat_c00.size(1), feat_c10.size(1), feat_c00.size(2)
        device = feat_c00.device

        # 1a. S_feat: 计算 TopicFormer 精炼特征间的相似度
        sim_feat = torch.einsum("nlc,nsc->nls", feat_c00, feat_c10) / C ** 0.5
        sim_feat = torch.clamp(sim_feat, min=-1e9, max=1e9)
        # 1b. S_topic: 计算主题概率间的相似度
        sim_topic = torch.ones(N, L, S, device=device)  # 默认

        if topic_matrix and 'img0' in topic_matrix and 'img1' in topic_matrix:
            try:
                topic_m0_prob = topic_matrix['img0']
                topic_m1_prob = topic_matrix['img1']
                K = topic_m0_prob.shape[-1]
                sim_topic = torch.einsum("nlk,nsk->nls", topic_m0_prob, topic_m1_prob)
            except Exception as e:
                pass
        sim_topic = torch.clamp(sim_topic, min=-1e9, max=1e9)

        # 1c. 归一化 S_feat
        norm_sim_feat = normalize_min_max(sim_feat, dim=(1, 2))
        # print(norm_sim_feat)
        # 1d. 融合: 残差门控 (Residual Gating)
        sim_matrix = norm_sim_feat * (1 + sim_topic)
        # sim_matrix = norm_sim_feat

        # 2. 根据匹配类型计算最终的置信度矩阵
        if self.match_type == 'dual_softmax':
            sim_matrix = sim_matrix / self.temperature
            if mask_c0 is not None and mask_c1 is not None:
                mask = mask_c0.unsqueeze(2) * mask_c1.unsqueeze(1)  # [N, L, S]
                sim_matrix.masked_fill_(~mask.bool(), -INF)
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        elif self.match_type == 'sinkhorn':
            if mask_c0 is not None and mask_c1 is not None:
                mask = mask_c0.unsqueeze(2) * mask_c1.unsqueeze(1)
                sim_matrix_masked = sim_matrix.clone()
                L_actual, S_actual = mask_c0.shape[1], mask_c1.shape[1]
                if sim_matrix.shape[1] == L_actual and sim_matrix.shape[2] == S_actual:
                    sim_matrix_masked.masked_fill_(~mask.bool(), -INF)
                else:
                    pass
                sim_matrix = sim_matrix_masked

            log_assign_matrix = self.log_optimal_transport(sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1]

            if not self.training and self.skh_prefilter:
                L_actual, S_actual = conf_matrix.shape[1], conf_matrix.shape[2]
                filter0 = (assign_matrix.max(dim=2)[1] == S_actual)[:, :L_actual]
                filter1 = (assign_matrix.max(dim=1)[1] == L_actual)[:, :S_actual]
                conf_matrix[filter0.unsqueeze(2).repeat(1, 1, S_actual)] = 0
                conf_matrix[filter1.unsqueeze(1).repeat(1, L_actual, 1)] = 0

            if self.config.get('sparse_spvs', False):
                data.update({'conf_matrix_with_bin': assign_matrix.clone()})
        conf_matrix = normalize_min_max(conf_matrix, dim=(1, 2))
        # 3. 将最终的置信度矩阵存储在 data 字典中
        data.update({'conf_matrix': conf_matrix})