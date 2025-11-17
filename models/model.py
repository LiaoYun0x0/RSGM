import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # 确保导入 numpy
from einops.einops import rearrange, repeat

# 假设这些导入路径在您的环境中是正确的
from common.functions import batch_get_mkpts
from configs.default import get_cfg_defaults, lower_config
from models.coarse_matching import CoarseMatching
from models.loftr_module.transformer import TopicFormer, FeatureFusion, Relator_Fusion
from models.resnet.swin_transformer import swin_transformer_with_FPN
from models.position import PositionEmbedding1D
from models.transformer import PositionEncodingSine, LocalFineFeatureTransformer
from models.resnet_fpn import ResNetFPN_8_2  # 导入 ResNetFPN

# local transformer parameters
cfg = {}
cfg["lo_cfg"] = {}
lo_cfg = cfg["lo_cfg"]
lo_cfg["d_model"] = 128
lo_cfg["layer_names"] = ["self", "cross"] * 1
lo_cfg["nhead"] = 8
lo_cfg["attention"] = "linear"

# --- 全局 config 加载 (保持不变) ---
config = get_cfg_defaults()
config = lower_config(config)
config = config["loftr"]


def _transform_inv(img, mean, std):
    # (这个函数在 model.py 中似乎未被使用，但保留)
    img = img * std + mean
    img = np.uint8(img * 255.0)
    img = img.transpose(1, 2, 0)
    return img


class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              stride=stride, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channel)
        self.mish = nn.Mish(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mish(x)
        return x


# ================== MatchingNet ==================
class MatchingNet(nn.Module):
    def __init__(
            self,
            d_coarse_model: int = 256,
            d_fine_model: int = 128,
            n_coarse_layers: int = 4,
            n_fine_layers: int = 1,
            n_heads: int = 8,
            backbone_name: str = 'resnet18',
            match_threshold: float = 0.2,
            window: int = 5,
            border: int = 1,
            sinkhorn_iterations: int = 100,
            matching_name: str = 'dual_softmax',  # 添加缺失的 matching_name
            **kwargs  # 【关键】添加 **kwargs 来接收所有其他参数 (如 use_sgm)
    ):
        super().__init__()

        # --- 【新增】将所有传入参数存储到 self.args 字典中 ---
        # 这样 forward 函数中的 getattr(self.args, ...) 才能工作
        self.args = locals().copy()
        self.args.update(kwargs)
        # ------------------------------------

        # --- 从传入参数中提取配置 ---
        config_base = get_cfg_defaults()
        config_base = lower_config(config_base)
        config_base = config_base["loftr"]

        # (使用传入的参数)
        window_size = window

        # 更新 Coarse Config (用于子模块)
        coarse_cfg = config_base['coarse']
        coarse_cfg['d_model'] = d_coarse_model
        coarse_cfg['nhead'] = n_heads
        coarse_cfg['layer_names_t'] = coarse_cfg.get('layer_names_t', ['seed', 'feat'] * 4)
        coarse_cfg['attention'] = coarse_cfg.get('attention', 'linear')
        coarse_cfg['temp_bug_fix'] = coarse_cfg.get('temp_bug_fix', False)
        # 【新增】传递超参数 K, alpha, beta
        coarse_cfg['neighbor_k'] = kwargs.get('neighbor_k', 3)
        coarse_cfg['fusion_alpha'] = kwargs.get('fusion_alpha', 0.5)
        coarse_cfg['fusion_beta'] = kwargs.get('fusion_beta', 0.5)

        # 更新 Fine Config (用于子模块)
        fine_cfg = config_base['fine']
        fine_cfg['d_model'] = d_fine_model
        fine_cfg['nhead'] = n_heads
        fine_cfg['layer_names'] = ["self", "cross"] * n_fine_layers
        fine_cfg['attention'] = coarse_cfg.get('attention', 'linear')

        # --- 【修复 R2.2】 粗匹配配置 (使用自适应阈值) ---
        adaptive_p = kwargs.get('adaptive_threshold_p', 0.8)  # 默认 0.8

        v1_match_cfg = {
            'adaptive_threshold_p': adaptive_p,  # 传递百分比
            'match_type': matching_name
        }
        # -----------------------------------------------

        # ================== 模块实例化 ==================
        self.backbone = swin_transformer_with_FPN()
        self.backbone_2 = ResNetFPN_8_2(config_base['resnetfpn'])

        self.pos_encoding = PositionEncodingSine(d_coarse_model, temp_bug_fix=coarse_cfg['temp_bug_fix'])
        self.position1d = PositionEmbedding1D(d_fine_model, max_len=window_size ** 2)

        self.local_transformer = LocalFineFeatureTransformer(fine_cfg)
        self.feature_fusion = FeatureFusion(coarse_cfg)  # 对应 PF/CNIM
        self.relator_fusion = Relator_Fusion(coarse_cfg)  # 对应 MM
        self.coarse_net = TopicFormer(coarse_cfg)  # 对应 SGM

        self.coarse_matching_v1_instance = CoarseMatching(v1_match_cfg)

        self.proj = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.merge = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.conv2d = ConvBN(d_fine_model, d_fine_model, 1, 1)
        self.regression1 = nn.Linear(d_coarse_model, d_fine_model, bias=True)

        # --- 【修复】硬编码 3200 ---
        regression2_in_dim = (window_size ** 2) * d_fine_model
        self.regression2 = nn.Linear(regression2_in_dim, d_fine_model, bias=True)
        # -------------------------

        self.regression = nn.Linear(d_fine_model, 2, bias=True)
        self.dropout = nn.Dropout(p=0.5)

        # --- 配置参数 ---
        self.border = border
        self.window = window_size
        self.match_threshold = match_threshold  # 奖励阈值 'u'
        self.step_coarse = 8
        self.step_fine = 2
        self.config_p = adaptive_p  # 存储自适应 p

    def fine_matching(self, x0, x1):
        x0, x1 = self.local_transformer(x0, x1)
        return x0, x1

    def _regression(self, feat):
        feat = self.regression1(feat)
        feat = feat.view(feat.shape[0], -1)  # [N_matches, W*W*C_fine]
        feat = self.dropout(feat)
        feat = self.regression2(feat)
        feat = self.regression(feat)
        return feat

    def unfold_within_window(self, featmap):
        # 【已修复】 您已将 stride 修复为 1
        stride = 1

        featmap_unfold = F.unfold(
            featmap,
            kernel_size=(self.window, self.window),
            stride=stride,
            padding=self.window // 2
        )
        featmap_unfold = rearrange(featmap_unfold, "B (C MM) L -> B L MM C", MM=self.window ** 2)
        return featmap_unfold

    def forward(self, samples0, samples1, gt_matrix=None):
        data = {}
        device = samples0.device

        # 1. 特征提取
        topic, feats_c, feats_f = self.backbone(torch.cat([samples0, samples1], dim=0))

        # 【新增 R2.5】 消融开关 (for Table 3)
        # (此处于 'model.py' 中被注释掉，与 'model_m.py' 逻辑对齐)
        # if self.args.get('use_backbone_2', True):  # 默认开启
        #     topic_2, feats_c_2, feats_f_2 = self.backbone_2(torch.cat([samples0, samples1], dim=0))
        #     topic = (topic + topic_2) / 2
        #     feats_c = (feats_c + feats_c_2) / 2
        #     feats_f = (feats_f + feats_f_2) / 2

        if topic.shape[0] % 2 != 0:
            raise ValueError(f"Batch size must be even (got {topic.shape[0]}). Running with B*2 images.")

        split_size = topic.shape[0] // 2
        (topic0, topic1) = topic.split(split_size)
        (mdesc0, mdesc1) = feats_c.split(split_size)
        (fine_featmap0, fine_featmap1) = feats_f.split(split_size)

        # 2. 粗特征 + 位置编码
        N, L, C = mdesc0.shape
        S = mdesc1.shape[1]

        # 【新增 R2.2】 为 retained_rate 计算 L
        num_total_pixels = torch.tensor(L, device=device, dtype=torch.float32)

        if L == 0 or S == 0 or (math.sqrt(L) % 1 != 0) or (math.sqrt(S) % 1 != 0):
            return {
                "cm_matrix": torch.empty(N, 0, 0, device=device),
                "mdesc0": mdesc0, "mdesc1": mdesc1,
                "mkpts1": torch.tensor([], device=device).view(0, 2),
                'mkpts0': torch.tensor([], device=device).view(0, 3),
                'samples0': samples0, 'samples1': samples1,
                'retained_rate': torch.tensor(0.0, device=device)  # 新增
            }

        H_c, W_c = int(math.sqrt(L)), int(math.sqrt(L))
        H_s, W_s = int(math.sqrt(S)), int(math.sqrt(S))

        # --- 【已修复】IndexError ---
        feat_c0_4d = rearrange(mdesc0, 'n (h w) c -> n c h w', h=H_c, w=W_c)
        feat_c1_4d = rearrange(mdesc1, 'n (h w) c -> n c h w', h=H_s, w=W_s)

        feat_c0_pos = self.pos_encoding(feat_c0_4d)
        feat_c1_pos = self.pos_encoding(feat_c1_4d)

        feat_c0 = rearrange(feat_c0_pos, 'n c h w -> n (h w) c')
        feat_c1 = rearrange(feat_c1_pos, 'n c h w -> n (h w) c')
        # -----------------------------------

        # --- Stage 1: 特征增强 (添加消融开关) ---

        # 3. 增强 A (FeatureFusion - PF/CNIM)
        if self.args.get('use_cnim', True):  # 默认开启 (R2.5)
            feat_ff0, feat_ff1 = self.feature_fusion(feat_c0, feat_c1, data)
        else:
            feat_ff0, feat_ff1 = feat_c0, feat_c1  # 跳过
            data['conf_matrix'] = torch.zeros(N, L, S, device=device)
            data['conf_matrix_fusion'] = torch.zeros(N, L, S, device=device)

        # 4. 增强 B (Relator_Fusion - MM)
        if self.args.get('use_mm', True):  # 默认开启 (R2.5)
            feat_rf0, feat_rf1 = self.relator_fusion(feat_ff0, feat_ff1, data)
        else:
            feat_rf0, feat_rf1 = feat_ff0, feat_ff1  # 跳过

        mask_c0 = mask_c1 = None

        # 3.1 调用 TopicFormer (SGM)
        if self.args.get('use_sgm', True):  # 默认开启 (R2.5)
            feat_c00, feat_c10, topic_matrix_match = self.coarse_net(
                feat_rf0, feat_rf1,
                topic0, topic1,
                mask_c0, mask_c1
            )
            data['topic_matrix_dict'] = topic_matrix_match
        else:
            feat_c00, feat_c10 = feat_rf0, feat_rf1  # 跳过
            topic_matrix_match = None

        # 3. V1 CoarseMatching
        self.coarse_matching_v1_instance(feat_c0, feat_c1, feat_c00, feat_c10, topic_matrix_match, data,
                                         mask_c0=mask_c0, mask_c1=mask_c1)
        cm_matrix_prob = data['conf_matrix']

        # --- 【修复 R2.2】: 手动实现过滤 (因为 CoarseMatching V1 不返回 matches) ---
        if 'matches' not in data:
            N_cm, L_cm, S_cm = cm_matrix_prob.shape
            p = self.config_p  # 使用 __init__ 中存储的 p
            min_val = cm_matrix_prob.view(N_cm, -1).min(dim=1)[0].view(N_cm, 1, 1)
            max_val = cm_matrix_prob.view(N_cm, -1).max(dim=1)[0].view(N_cm, 1, 1)
            theta_c = min_val + p * (max_val - min_val)
            mask_thr = cm_matrix_prob > theta_c
            mask_mnn = (cm_matrix_prob == cm_matrix_prob.max(dim=2, keepdim=True)[0]) \
                       * (cm_matrix_prob == cm_matrix_prob.max(dim=1, keepdim=True)[0])
            cf_matrix = cm_matrix_prob * mask_thr * mask_mnn
            mask_v, all_j_ids = cf_matrix.max(dim=2)
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
            matches = torch.stack([b_ids, i_ids, j_ids]).T
            data['matches'] = matches

        matches = data['matches']
        mdesc0 = feat_c00
        mdesc1 = feat_c10
        # ----------------------------------------------------

        # 4. 精细匹配准备
        N_f, L_f, C_f = fine_featmap0.shape
        if L_f == 0 or (math.sqrt(L_f) % 1 != 0):
            return {
                "cm_matrix": cm_matrix_prob,
                "mdesc0": mdesc0, "mdesc1": mdesc1,
                "mkpts1": torch.tensor([], device=device).view(0, 2),
                'mkpts0': torch.tensor([], device=device).view(0, 3),
                'samples0': samples0, 'samples1': samples1,
                'retained_rate': torch.tensor(0.0, device=device)
            }

        H_f, W_f = int(math.sqrt(L_f)), int(math.sqrt(L_f))
        fine_featmap0_4d = rearrange(fine_featmap0, 'n (h w) c -> n c h w', h=H_f, w=W_f)
        fine_featmap1_4d = rearrange(fine_featmap1, 'n (h w) c -> n c h w', h=H_f, w=W_f)

        fine_featmap0 = self.conv2d(fine_featmap0_4d)
        fine_featmap1 = self.conv2d(fine_featmap1_4d)

        # --- 【新增 R2.2】: 计算保留率 ---
        num_retained = torch.tensor(matches.shape[0], device=device, dtype=torch.float32)
        retained_rate = num_retained / num_total_pixels if num_total_pixels > 0 else torch.tensor(0.0, device=device)
        # --------------------------------

        # 5. 空匹配检查
        if matches.shape[0] == 0:
            return {
                'cm_matrix': cm_matrix_prob,
                "mdesc0": mdesc0,
                "mdesc1": mdesc1,
                "mkpts1": torch.tensor([], device=device).view(0, 2),
                'mkpts0': torch.tensor([], device=device).view(0, 3),
                'samples0': samples0,
                'samples1': samples1,
                'retained_rate': retained_rate
            }

        # 6. 精细匹配流程
        mkpts0, mkpts1 = batch_get_mkpts(matches, samples0, samples1, patch_size=self.step_coarse)

        fine_featmap0_unfold = self.unfold_within_window(fine_featmap0)
        fine_featmap1_unfold = self.unfold_within_window(fine_featmap1)

        i_ids_clamped = matches[:, 1].clamp(max=fine_featmap0_unfold.shape[1] - 1)
        j_ids_clamped = matches[:, 2].clamp(max=fine_featmap1_unfold.shape[1] - 1)

        local_desc = torch.cat([
            fine_featmap0_unfold[matches[:, 0], i_ids_clamped],
            fine_featmap1_unfold[matches[:, 0], j_ids_clamped]
        ], dim=0)

        center_desc = repeat(torch.cat([
            mdesc0[matches[:, 0], matches[:, 1]],
            mdesc1[matches[:, 0], matches[:, 2]]
        ], dim=0), 'N C -> N WW C', WW=self.window ** 2)

        center_desc = self.proj(center_desc)
        local_desc = torch.cat([local_desc, center_desc], dim=-1)
        local_desc = self.merge(local_desc)

        local_position = self.position1d(local_desc)
        local_desc = local_desc + local_position

        desc0, desc1 = torch.chunk(local_desc, 2, dim=0)
        fdesc0, fdesc1 = self.fine_matching(desc0, desc1)

        c = self.window ** 2 // 2
        center_desc = repeat(fdesc0[:, c, :], 'N C->N WW C', WW=self.window ** 2)
        center_desc = torch.cat([center_desc, fdesc1], dim=-1)
        expected_coords = self._regression(center_desc)
        mkpts1 = mkpts1[:, 1:] + expected_coords

        return {
            'cm_matrix': cm_matrix_prob,
            'matches': matches,
            'samples0': samples0,
            'samples1': samples1,
            'mkpts1': mkpts1,
            'mkpts0': mkpts0,
            'mdesc0': mdesc0,
            'mdesc1': mdesc1,
            'topic_matrix': data['topic_matrix_dict'],
            'retained_rate': retained_rate  # 【新增】
        }