import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from common.functions import batch_get_mkpts
from configs.default import get_cfg_defaults, lower_config
from models.coarse_matching import CoarseMatching

from models.loftr_module.transformer import TopicFormer, FeatureFusion, Relator_Fusion
from models.resnet.swin_transformer import swin_transformer_with_FPN
from models.position import PositionEmbedding1D
from models.transformer import PositionEncodingSine, LocalFineFeatureTransformer


# ================== ConvBN ==================
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
            matching_name: str = 'sinkhorn',
            match_threshold: float = 0.2,
            window: int = 5,
            border: int = 1,
            sinkhorn_iterations: int = 100,
    ):
        super().__init__()

        # --- 配置修复 & 初始化 TopicFormer ---
        config_base = get_cfg_defaults()
        config_base = lower_config(config_base)
        config_base = config_base["loftr"]

        # --- Coarse Config ---
        coarse_cfg = config_base['coarse']
        coarse_cfg['d_model'] = d_coarse_model
        coarse_cfg['nhead'] = n_heads
        coarse_cfg['n_layers'] = n_coarse_layers

        # ✅ 修复 KeyError
        coarse_cfg['layer_names_t'] = ['seed'] * (n_coarse_layers // 2) + ['seed', 'feat'] * (n_coarse_layers // 2)
        coarse_cfg['attention'] = coarse_cfg.get('attention', 'linear')
        coarse_cfg['temp_bug_fix'] = coarse_cfg.get('temp_bug_fix', False)

        # --- Fine Config ---
        fine_cfg = {}
        fine_cfg['d_model'] = d_fine_model
        fine_cfg['nhead'] = n_heads
        fine_cfg['layer_names'] = ["self", "cross"] * n_fine_layers
        fine_cfg['attention'] = coarse_cfg['attention']

        # --- V1 粗匹配配置 ---
        v1_match_cfg = {'thr': match_threshold}

        # ================== 模块实例化 ==================
        self.backbone = swin_transformer_with_FPN()
        self.pos_encoding = PositionEncodingSine(d_coarse_model, temp_bug_fix=coarse_cfg['temp_bug_fix'])
        self.position1d = PositionEmbedding1D(d_fine_model, max_len=window ** 2)

        self.local_transformer = LocalFineFeatureTransformer(fine_cfg)
        # Stage 1: 特征增强
        self.feature_fusion = FeatureFusion(coarse_cfg)
        self.relator_fusion = Relator_Fusion(coarse_cfg)
        self.coarse_net = TopicFormer(coarse_cfg)

        self.coarse_matching_v1_instance = CoarseMatching(v1_match_cfg)

        self.proj = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.merge = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.conv2d = ConvBN(d_fine_model, d_fine_model, 1, 1)
        self.regression1 = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        regression2_in_dim = (window ** 2) * d_fine_model
        self.regression2 = nn.Linear(regression2_in_dim, d_fine_model, bias=True)
        self.regression = nn.Linear(d_fine_model, 2, bias=True)
        self.dropout = nn.Dropout(p=0.5)

        # --- 配置参数 ---
        self.border = border
        self.window = window
        self.match_threshold = match_threshold
        self.step_coarse = 8
        self.step_fine = 2
        self.config_p = match_threshold

    def fine_matching(self, x0, x1):
        x0, x1 = self.local_transformer(x0, x1)
        return x0, x1

    def _regression(self, feat):
        feat = self.regression1(feat)
        feat = feat.view(feat.shape[0], -1)
        feat = self.dropout(feat)
        feat = self.regression2(feat)
        feat = self.regression(feat)
        return feat

    def unfold_within_window(self, featmap):
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
        split_size = topic.shape[0] // 2
        (topic0, topic1) = topic.split(split_size)
        (mdesc0, mdesc1) = feats_c.split(split_size)
        (fine_featmap0, fine_featmap1) = feats_f.split(split_size)

        # 2. 粗特征 + 位置编码
        N, L, C = mdesc0.shape
        S = mdesc1.shape[1]

        if L == 0 or S == 0 or (math.sqrt(L) % 1 != 0) or (math.sqrt(S) % 1 != 0):
            return {
                "cm_matrix": torch.empty(N, 0, 0, device=device),
                "mdesc0": mdesc0, "mdesc1": mdesc1,
                "mkpts1": torch.tensor([], device=device).view(0, 2),
                'mkpts0': torch.tensor([], device=device).view(0, 3),
                'samples0': samples0, 'samples1': samples1
            }

        H_c, W_c = int(math.sqrt(L)), int(math.sqrt(L))
        H_s, W_s = int(math.sqrt(S)), int(math.sqrt(S))

        feat_c0_4d = rearrange(mdesc0, 'n (h w) c -> n c h w', h=H_c, w=W_c)
        feat_c1_4d = rearrange(mdesc1, 'n (h w) c -> n c h w', h=H_s, w=W_s)

        feat_c0_pos = self.pos_encoding(feat_c0_4d)
        feat_c1_pos = self.pos_encoding(feat_c1_4d)

        feat_c0 = rearrange(feat_c0_pos, 'n c h w -> n (h w) c')
        feat_c1 = rearrange(feat_c1_pos, 'n c h w -> n (h w) c')

        # --- Stage 1: 特征增强 ---

        # 3. 增强 A (FeatureFusion)
        # 输入: [N, L, 256] -> 输出: [N, L, 256]
        feat_ff0, feat_ff1 = self.feature_fusion(
            feat_c0, feat_c1, data
        )

        # 4. 增强 B (Relator_Fusion)
        # 输入: [N, L, 256] -> 输出: [N, L, 256]
        feat_rf0, feat_rf1 = self.relator_fusion(
            feat_ff0, feat_ff1, data
        )

        mask_c0 = mask_c1 = None
        # 3.1 调用 TopicFormer (self.coarse_net)
        feat_c00, feat_c10, topic_matrix_match = self.coarse_net(
            feat_rf0, feat_rf1,
            topic0, topic1,
            mask_c0, mask_c1
        )
        data['topic_matrix_dict'] = topic_matrix_match
        # 3. V1 CoarseMatching
        # self.coarse_matching_v1_instance(feat_c00, feat_c10, data, mask_c0=mask_c0, mask_c1=mask_c1)
        self.coarse_matching_v1_instance(feat_c0,feat_c1,feat_c00, feat_c10,topic_matrix_match, data, mask_c0=mask_c0, mask_c1=mask_c1)
        cm_matrix_prob = data['conf_matrix']
        matches = data['matches']
        mdesc0 = feat_c00
        mdesc1 = feat_c10

        # 4. 精细匹配准备
        N_f, L_f, C_f = fine_featmap0.shape
        if L_f == 0 or (math.sqrt(L_f) % 1 != 0):
            return {
                "cm_matrix": cm_matrix_prob,
                "mdesc0": mdesc0, "mdesc1": mdesc1,
                "mkpts1": torch.tensor([], device=device).view(0, 2),
                'mkpts0': torch.tensor([], device=device).view(0, 3),
                'samples0': samples0, 'samples1': samples1
            }

        H_f, W_f = int(math.sqrt(L_f)), int(math.sqrt(L_f))
        fine_featmap0_4d = rearrange(fine_featmap0, 'n (h w) c -> n c h w', h=H_f, w=W_f)
        fine_featmap1_4d = rearrange(fine_featmap1, 'n (h w) c -> n c h w', h=H_f, w=W_f)

        fine_featmap0 = self.conv2d(fine_featmap0_4d)
        fine_featmap1 = self.conv2d(fine_featmap1_4d)

        # 5. 空匹配检查
        if matches.shape[0] == 0:
            return {
                'cm_matrix': cm_matrix_prob,
                "mdesc0": mdesc0,
                "mdesc1": mdesc1,
                "mkpts1": torch.tensor([], device=device).view(0, 2),
                'mkpts0': torch.tensor([], device=device).view(0, 3),
                'samples0': samples0,
                'samples1': samples1
            }

        # 6. 精细匹配流程
        mkpts0, mkpts1 = batch_get_mkpts(matches, samples0, samples1, patch_size=self.step_coarse)
        fine_featmap0_unfold = self.unfold_within_window(fine_featmap0)
        fine_featmap1_unfold = self.unfold_within_window(fine_featmap1)

        local_desc = torch.cat([
            fine_featmap0_unfold[matches[:, 0], matches[:, 1]],
            fine_featmap1_unfold[matches[:, 0], matches[:, 2]]
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
            'topic_matrix':data['topic_matrix_dict'],
        }
