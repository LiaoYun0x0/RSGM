import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops.einops import rearrange, repeat

from common.functions import batch_get_mkpts
# (假设 batch_get_mkpts 在其他地方定义)
# from common.functions import * def batch_get_mkpts(matches, samples0, samples1):
# 占位符：
# mkpts0 = torch.rand(matches.shape[0], 3, device=matches.device)  # (b_idx, x, y)
# mkpts1 = torch.rand(matches.shape[0], 3, device=matches.device)  # (b_idx, x, y)
# mkpts0[:, 0] = matches[:, 0].float()
# mkpts1[:, 0] = matches[:, 0].float()
# return mkpts0, mkpts1

# 导入重构后的 CoarseMatching
from .coarse_matching import CoarseMatching

# 导入新的依赖项
from configs.default import get_cfg_defaults, lower_config

# (假设这些模块位于你指定的路径)
try:
    from models.loftr_module.transformer import LocalFeatureTransformer as LocalTransformer, TopicFormer
    from models.resnet.swin_transformer import swin_transformer_with_FPN
    from models.transformer import LocalFineFeatureTransformer, PositionEncodingSine
    from models.position import PositionEmbedding1D  # 保留1D位置编码
except ImportError:
    print("Warning: Could not import all modules. Using placeholders.")


    # 定义占位符以便脚本能被解析
    class PlaceholderModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            if len(args) == 0: return None
            if len(args) == 1: return args[0]
            if len(args) == 2: return args[0], args[1]
            if len(args) == 3: return args[0], args[1], args[2]
            return args


    SwinTransformerWithFPN = PlaceholderModule
    TopicFormer = PlaceholderModule
    LocalFineFeatureTransformer = PlaceholderModule
    PositionEncodingSine = PlaceholderModule
    PositionEmbedding1D = PlaceholderModule


class ConvBN(nn.Module):
    # (保持不变)
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channel)
        self.mish = nn.Mish(inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.mish(x)
        return x


class MatchingNet(nn.Module):
    def __init__(
            self,
            d_coarse_model: int = 256,
            d_fine_model: int = 128,
            n_coarse_layers: int = 4,  # (使用 V1 脚本传入的值)
            n_fine_layers: int = 1,  # (使用 V1 脚本传入的值)
            n_heads: int = 8,  # (使用 V1 脚本传入的值)
            backbone_name: str = 'resnet18',
            matching_name: str = 'sinkhorn',
            match_threshold: float = 0.2,
            window: int = 5,
            border: int = 1,
            sinkhorn_iterations: int = 100,  # (使用 V1 脚本传入的值)
    ):
        super().__init__()

        # --- A.1: 加载 V2 基础配置 ---
        # 我们仍然需要它来获取 'attention', 'match_type' 等 V2 特有的设置
        config_base = get_cfg_defaults()
        config_base = lower_config(config_base)
        config_base = config_base["loftr"]

        # --- A.2: 【修复】用 V1 脚本参数覆盖 V2 基础配置 ---

        # 1. Coarse Config (用于 TopicFormer, PositionEncoding)
        coarse_cfg = config_base['coarse']
        coarse_cfg['d_model'] = d_coarse_model
        coarse_cfg['nhead'] = n_heads
        # (假设 n_coarse_layers 对应 TopicFormer 的层数)
        # (你可能需要检查 TopicFormer 的 'layer_names_t' 是否与 n_coarse_layers 匹配)

        # 2. Fine Config (用于 LocalFineFeatureTransformer)
        fine_cfg = {}
        fine_cfg['d_model'] = d_fine_model
        fine_cfg['nhead'] = n_heads
        fine_cfg['layer_names'] = ["self", "cross"] * n_fine_layers  # 动态设置层数
        fine_cfg['attention'] = config_base['coarse']['attention']  # 沿用 'linear'

        # 3. Match Config (用于 CoarseMatching)
        match_cfg = config_base['match_coarse']
        match_cfg['match_type'] = matching_name
        match_cfg['skh_iters'] = sinkhorn_iterations
        match_cfg['thr'] = match_threshold

        # --- 修复结束 ---

        # 1. 主干网络
        self.backbone = swin_transformer_with_FPN()

        # 2. 位置编码 (使用 d_coarse_model)
        self.pos_encoding = PositionEncodingSine(
            d_coarse_model,  # <-- 关键修复
            temp_bug_fix=coarse_cfg['temp_bug_fix'])

        self.position1d = PositionEmbedding1D(d_fine_model, max_len=window ** 2)

        # 3. 实例化 V2 核心模块 (使用更新后的配置)
        self.local_transformer = LocalFineFeatureTransformer(fine_cfg)
        self.coarse_net = TopicFormer(coarse_cfg)

        # 4. 实例化 V2 粗匹配模块
        self.coarse_matching = CoarseMatching(match_cfg)

        # 5. 精细匹配所需的层 (使用 d_coarse_model 和 d_fine_model)
        self.proj = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.merge = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.conv2d = ConvBN(d_fine_model, d_fine_model, 1, 1)

        self.regression1 = nn.Linear(d_coarse_model, d_fine_model, bias=True)

        # 动态计算回归层维度
        regression2_in_dim = (window ** 2) * d_fine_model
        self.regression2 = nn.Linear(regression2_in_dim, d_fine_model, bias=True)
        self.regression = nn.Linear(d_fine_model, 2, bias=True)
        self.dropout = nn.Dropout(p=0.5)

        # 6. 保留配置参数
        self.border = border
        self.window = window
        self.match_threshold = match_threshold
        self.step_coarse = 8
        self.step_fine = 2

        self.th = 0.1
        self.config_p = match_cfg.get('adaptive_threshold_p', 0.8)

    def fine_matching(self, x0, x1):
        # (保持不变)
        x0, x1 = self.local_transformer(x0, x1)
        return x0, x1

    def _regression(self, feat):
        # (保持不变)
        feat = self.regression1(feat)
        feat = feat.view(feat.shape[0], -1)
        feat = self.dropout(feat)
        feat = self.regression2(feat)
        feat = self.regression(feat)
        return feat

    def unfold_within_window(self, featmap):
        # (保持不变)
        scale = self.step_coarse - self.step_fine
        stride = 1

        featmap_unfold = F.unfold(
            featmap,
            kernel_size=(self.window, self.window),
            stride=stride,
            padding=self.window // 2
        )

        featmap_unfold = rearrange(
            featmap_unfold,
            "B (C MM) L -> B L MM C",
            MM=self.window ** 2
        )
        return featmap_unfold

    def forward(self, samples0, samples1, gt_matrix):

        data = {}
        device = samples0.device

        # --- 1. 特征提取 (仅Swin) ---
        topic, feats_c, feats_f = self.backbone(torch.cat([samples0, samples1], dim=0))

        # --- 1.5. 分离 Batch ---
        split_size = topic.shape[0] // 2
        (topic0, topic1) = topic.split(split_size)
        (mdesc0, mdesc1) = feats_c.split(split_size)  # [N, L, C] 粗特征
        (fine_featmap0, fine_featmap1) = feats_f.split(split_size)  # [N, L, C] 精特征

        # --- 2. 粗特征 + 位置编码 ---
        N, L, C = mdesc0.shape
        S = mdesc1.shape[1]

        # (增加一个检查，以防 L=0)
        if L == 0 or S == 0:
            # print("Warning: Empty feature map detected.")
            return {
                "cm_matrix": torch.empty(N, 0, 0, device=device),
                "mdesc0": mdesc0, "mdesc1": mdesc1,
                "mkpts1": torch.tensor([], device=device).view(0, 2),
                'mkpts0': torch.tensor([], device=device).view(0, 3),
                'samples0': samples0, 'samples1': samples1
            }

        H_c = int(math.sqrt(L))
        W_c = int(math.sqrt(L))
        H_s = int(math.sqrt(S))
        W_s = int(math.sqrt(S))

        # (增加一个检查，防止 L 不是完美平方数)
        if H_c * W_c != L or H_s * W_s != S:
            # print(f"Warning: Non-square feature map L={L}, S={S}. Skipping batch.")
            # (这通常在输入图像尺寸可变时发生，这里我们先返回空)
            return {
                "cm_matrix": torch.empty(N, 0, 0, device=device),
                "mdesc0": mdesc0, "mdesc1": mdesc1,
                "mkpts1": torch.tensor([], device=device).view(0, 2),
                'mkpts0': torch.tensor([], device=device).view(0, 3),
                'samples0': samples0, 'samples1': samples1
            }

        feat_c0_4d = rearrange(mdesc0, 'n (h w) c -> n c h w', h=H_c, w=W_c)
        feat_c1_4d = rearrange(mdesc1, 'n (h w) c -> n c h w', h=H_s, w=W_s)

        # (这是之前崩溃的地方)
        feat_c0_pos = self.pos_encoding(feat_c0_4d)
        feat_c1_pos = self.pos_encoding(feat_c1_4d)

        feat_c0 = rearrange(feat_c0_pos, 'n c h w -> n (h w) c')
        feat_c1 = rearrange(feat_c1_pos, 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None

        # --- 3. 粗匹配 ---
        feat_c00, feat_c10, topic_matrix_match = self.coarse_net(feat_c0, feat_c1, topic0,
                                                                 topic1, mask_c0,
                                                                 mask_c1)

        self.coarse_matching(feat_c0, feat_c1, feat_c00, feat_c10, topic_matrix_match, data,
                             mask_c0=mask_c0, mask_c1=mask_c1)

        cm_matrix = data['conf_matrix']
        mdesc0 = feat_c00
        mdesc1 = feat_c10

        # --- 4. 精细特征准备 ---
        N_f, L_f, C_f = fine_featmap0.shape
        H_f = int(math.sqrt(L_f))
        W_f = int(math.sqrt(L_f))

        fine_featmap0_4d = rearrange(fine_featmap0, 'n (h w) c -> n c h w', h=H_f, w=W_f)
        fine_featmap1_4d = rearrange(fine_featmap1, 'n (h w) c -> n c h w', h=H_f, w=W_f)

        fine_featmap0 = self.conv2d(fine_featmap0_4d)
        fine_featmap1 = self.conv2d(fine_featmap1_4d)

        # --- 5. 提取粗匹配对 ---
        N, L, S = cm_matrix.shape

        min_val = cm_matrix.view(N, -1).min(dim=1)[0].view(N, 1, 1)
        max_val = cm_matrix.view(N, -1).max(dim=1)[0].view(N, 1, 1)
        theta_c = min_val + self.config_p * (max_val - min_val)

        mask_mnn = (cm_matrix == cm_matrix.max(dim=2, keepdim=True)[0]) \
                   * (cm_matrix == cm_matrix.max(dim=1, keepdim=True)[0])

        mask_thr = cm_matrix > theta_c
        cf_matrix = cm_matrix * mask_thr * mask_mnn

        mask_v, all_j_ids = cf_matrix.max(dim=2)
        b_ids, i_ids = torch.where(mask_v > 0)
        j_ids = all_j_ids[b_ids, i_ids]
        matches = torch.stack([b_ids, i_ids, j_ids]).T

        # --- 6. 检查空匹配 ---
        if matches.shape[0] == 0:
            return {
                "cm_matrix": cm_matrix,
                "mdesc0": mdesc0,
                "mdesc1": mdesc1,
                "mkpts1": torch.tensor([], device=device).view(0, 2),
                'mkpts0': torch.tensor([], device=device).view(0, 3),
                'samples0': samples0,
                'samples1': samples1
            }

        # --- 7. 精细匹配流程 ---
        mkpts0, mkpts1 = batch_get_mkpts(matches, samples0, samples1)
        fine_featmap0_unfold = self.unfold_within_window(fine_featmap0)
        fine_featmap1_unfold = self.unfold_within_window(fine_featmap1)

        local_desc = torch.cat([
            fine_featmap0_unfold[matches[:, 0], matches[:, 1]],
            fine_featmap1_unfold[matches[:, 0], matches[:, 2]]
        ], dim=0)

        center_desc = repeat(torch.cat([
            mdesc0[matches[:, 0], matches[:, 1]],
            mdesc1[matches[:, 0], matches[:, 2]]
        ], dim=0),
            'N C -> N WW C',
            WW=self.window ** 2)

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
            'cm_matrix': cm_matrix,
            'matches': matches,
            'samples0': samples0,
            'samples1': samples1,
            'mkpts1': mkpts1,  # (n, 2)
            'mkpts0': mkpts0,  # (n, 3) (b_idx, x, y)
            'mdesc0': mdesc0,
            'mdesc1': mdesc1,
        }