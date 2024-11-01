import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat
from kornia.utils import create_meshgrid

from configs.default import get_cfg_defaults, lower_config

INF = 1e9
config = get_cfg_defaults()
config = lower_config(config)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


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


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']
        self.head = 8
        self.global_avg_pool1 = nn.AdaptiveAvgPool2d((10, 10))
        self.global_avg_pool2 = nn.AdaptiveAvgPool2d((5, 5))
        # self.global_avg_pool3 = nn.AdaptiveAvgPool2d((5, 5))
        self.downt3 = conv1x1(1100, 128)
        self.downt4 = conv1x1(256, 128)
        self.down1_16 = conv1x1(768, 256)
        self.down1_2 = conv1x1(196 * 2, 196)
        self.down1_4 = conv1x1(512, 256)
        self.down1_8 = conv1x1(512, 256)
        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        if self.match_type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        elif self.match_type == 'sinkhorn':
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(
                torch.tensor(config['skh_init_bin_score'], requires_grad=True))
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
        else:
            raise NotImplementedError()

    def forward(self, feats12_0, feats12_1, feats14_0, feats14_1, feats18_0, feats18_1, feats116_0, feats116_1, t1, t2,
                t3, t4, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        # spvs_coarse(data, config)
        L = 1600
        # print(feats1_0.shape)
        conf_matrix_116 = torch.einsum("nlc,nsc->nls", feats116_0, feats116_1)
        conf_matrix_116 = F.softmax(conf_matrix_116, 1) * F.softmax(conf_matrix_116, 2)
        conf_matrix_116_idx = torch.argmax(conf_matrix_116, -1)
        feats116_1 = feats116_1[:, conf_matrix_116_idx[-1]]
        feats116 = torch.reshape(torch.concat((feats116_0, feats116_1), -1),
                                 (feats116_0.shape[0], 20, 20, feats116_0.shape[2] * 2)).permute(0, 3, 1, 2)

        t2_ = torch.reshape(t2, (t2.shape[0], 20, 20, t1.shape[2])).permute(0, 3, 1, 2)
        t2_ = self.down1_16(torch.concat((t2_, feats116), 1))  # 1,256,40,40
        # t2_ = (t2_ + feats116) / 2
        # print(t2_.shape)
        # t1_ = self.global_avg_pool2(t1_)  # 1,256,20,20
        t2 = torch.reshape(t2_, (t2_.shape[0], t2_.shape[1], t2_.shape[2] * t2_.shape[3])).permute(0, 2, 1)
        # feats18 = torch.concat((feats18_0, feats18_1), 1)
        # dmatrix18 = torch.einsum("nmd,nkd->nmk", feats18, t2)
        conf_matrix_18 = torch.einsum("nmd,nkd->nmk", feats18_0, feats18_1)
        # feat_topics18 = torch.zeros_like(dmatrix18).scatter_(-1, torch.argmax(dmatrix18, dim=-1, keepdim=True), 1.0)
        # print(feat_topics14.shape)
        # topic_matrix_img0 = feat_topics18[:, :L].transpose(1, 2)  # [B,S,L]
        # topic_matrix_img1 = feat_topics18[:, L:].transpose(1, 2)
        # topic_matrix_img0 = torch.einsum("nml,nld->nmld", topic_matrix_img0, feats18_0)
        # N, M, L, C = topic_matrix_img0.shape
        # topic_matrix_img1 = torch.einsum("nml,nld->nmld", topic_matrix_img1, feats18_1)
        # conf_matrix_18 = torch.einsum("nmhd,nlsd->nhs", topic_matrix_img0, topic_matrix_img1)
        # print(topic_matrix_img0.shape)
        # print(topic_matrix_img1.shape)
        # print(conf_matrix_18.shape)
        conf_matrix_18 = F.softmax(conf_matrix_18, 1) * F.softmax(conf_matrix_18, 2)
        conf_matrix_18_idx = torch.argmax(conf_matrix_18, -1)
        feats18_1 = feats18_1[:, conf_matrix_18_idx[-1]]
        feats18 = torch.reshape(torch.concat((feats18_0, feats18_1), -1),
                                (feats18_0.shape[0], 40, 40, feats18_0.shape[2] * 2)).permute(0, 3, 1, 2)
        feats18 = self.down1_8(feats18)  # 1,196,80,80
        feats18 = self.global_avg_pool1(feats18)  # 1,256,20,20
        t2 = self.global_avg_pool1(t2_)  # 1,256,20,20
        t3 = torch.reshape(t3, (t3.shape[0], 40 // 4, 40 // 4, t3.shape[2])).permute(0, 3, 1, 2)

        conf_matrix_14 = torch.einsum("nlc,nsc->nls", feats14_0, feats14_1)
        conf_matrix_14 = F.softmax(conf_matrix_14, 1) * F.softmax(conf_matrix_14, 2)
        conf_matrix_14_idx = torch.argmax(conf_matrix_14, -1)
        feats14_1 = feats14_1[:, conf_matrix_14_idx[-1]]
        feats14 = torch.reshape(torch.concat((feats14_0, feats14_1), -1),
                                (feats14_0.shape[0], 80, 80, feats14_0.shape[2] * 2)).permute(0, 3, 1, 2)
        feats14 = self.global_avg_pool1(feats14)  # 1,256,20,20
        t3 = torch.concat((t3, t2, feats18, feats14), 1)
        t3 = self.downt3(t3)
        t3 = self.global_avg_pool2(t3)
        # print(t3.shape)
        t4 = torch.reshape(t4, (t4.shape[0], 40 // 8, 40 // 8, t4.shape[2])).permute(0, 3, 1, 2)
        t4 = torch.concat((t3, t4), 1)
        t4 = self.downt4(t4)
        # print(t4.shape)
        t4 = torch.reshape(t4, (t4.shape[0], t4.shape[1], t4.shape[2] * t4.shape[3])).permute(0, 2, 1)
        L = 160 * 160
        feats12 = torch.concat((feats12_0, feats12_1), 1)
        dmatrix12 = torch.einsum("nmd,nkd->nmk", feats12, t4)
        feat_topics12 = torch.zeros_like(dmatrix12).scatter_(-1, torch.argmax(dmatrix12, dim=-1, keepdim=True), 1.0)
        # print(feat_topics14.shape)
        topic_matrix_img0 = feat_topics12[:, :L].transpose(1, 2)  # [B,S,L]
        topic_matrix_img1 = feat_topics12[:, L:].transpose(1, 2)
        topic_matrix_img0 = torch.einsum("nml,nld->nmld", topic_matrix_img0, feats12_0)
        N, M, L, C = topic_matrix_img0.shape
        topic_matrix_img1 = torch.einsum("nml,nld->nmld", topic_matrix_img1, feats12_1)
        # topic_matrix_img0 = torch.reshape(topic_matrix_img0, (N, L, self.head, (C // self.head) * M))
        # topic_matrix_img1 = torch.reshape(topic_matrix_img1, (N, L, self.head, (C // self.head) * M))
        conf_matrix_12 = torch.einsum("nmhd,nlsd->nhs", topic_matrix_img0, topic_matrix_img1)
        # conf_matrix_12 = torch.mean(conf_matrix_12, -1)
        # print(conf_matrix_12.shape)
        conf_matrix = conf_matrix_18
        conf_matrix_12 = F.softmax(conf_matrix_12, 1) * F.softmax(conf_matrix_12, 2) / self.temperature
        conf_matrix_12_mask = (conf_matrix_12 > 1).byte()
        conf_matrix_12 = conf_matrix_12 * conf_matrix_12_mask
        index_i = torch.nonzero(conf_matrix_12, as_tuple=False)
        index_j = torch.nonzero(conf_matrix_12.permute(0, 2, 1), as_tuple=False)
        # print(index_i)
        # print(index_j)
        # conf_matrix_12_idx = torch.argmax(conf_matrix_12, -1)[-1]
        # print(torch.sum(conf_matrix_12,-1).shape)
        # print(conf_matrix_12_idx.shape)
        # print(conf_matrix_12)
        # conf_matrix_12_jdx = torch.argmax(conf_matrix_12, -2)[-1]
        # print(conf_matrix_12_idx.shape)
        # feats12_1 = feats12_1[:, conf_matrix_12_idx[-1]]
        # feats12 = torch.reshape(torch.concat((feats12_0, feats12_1), -1),
        #                          (feats12_0.shape[0], 160, 160, feats12_0.shape[2] * 2)).permute(0, 3, 1, 2)
        # data.update({'t3': t3})
        data.update({'conf_matrix': conf_matrix})
        data.update({'index_i': index_i})
        data.update({'index_j': index_j})

        # predict coarse matches from conf_matrix
        # data.update({'matches': self.get_coarse_match(conf_matrix, data)})

#     @torch.no_grad()
#     def get_coarse_match(self, conf_matrix, data):
#         """
#         Args:
#             conf_matrix (torch.Tensor): [N, L, S]
#             data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
#         Returns:
#             coarse_matches (dict): {
#                 'b_ids' (torch.Tensor): [M'],
#                 'i_ids' (torch.Tensor): [M'],
#                 'j_ids' (torch.Tensor): [M'],
#                 'gt_mask' (torch.Tensor): [M'],
#                 'm_bids' (torch.Tensor): [M],
#                 'mkpts0_c' (torch.Tensor): [M, 2],
#                 'mkpts1_c' (torch.Tensor): [M, 2],
#                 'mconf' (torch.Tensor): [M]}
#         """
#         axes_lengths = {
#             'h0c': 40,
#             'w0c': 40,
#             'h1c': 40,
#             'w1c': 40
#         }
#         _device = conf_matrix.device
#         # 1. confidence thresholding
#         mask = conf_matrix > self.thr
#         mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
#                          **axes_lengths)
#         if 'mask0' not in data:
#             mask_border(mask, self.border_rm, False)
#         else:
#             mask_border_with_padding(mask, self.border_rm, False,
#                                      data['mask0'], data['mask1'])
#         mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
#                          **axes_lengths)
#
#         # 2. mutual nearest
#         mask = mask \
#                * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
#                * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
#
#         # 3. find all valid coarse matches
#         # this only works when at most one `True` in each row
#         mask_v, all_j_ids = mask.max(dim=2)
#         b_ids, i_ids = torch.where(mask_v)
#         j_ids = all_j_ids[b_ids, i_ids]
#         mconf = conf_matrix[b_ids, i_ids, j_ids]
#
#         # 4. Random sampling of training samples for fine-level LoFTR
#         # (optional) pad samples with gt coarse-level matches
#         if self.training:
#             # NOTE:
#             # The sampling is performed across all pairs in a batch without manually balancing
#             # #samples for fine-level increases w.r.t. batch_size
#             if 'mask0' not in data:
#                 num_candidates_max = mask.size(0) * max(
#                     mask.size(1), mask.size(2))
#             else:
#                 num_candidates_max = compute_max_candidates(
#                     data['mask0'], data['mask1'])
#             num_matches_train = int(num_candidates_max *
#                                     self.train_coarse_percent)
#             num_matches_pred = len(b_ids)
#             assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"
#
#             # pred_indices is to select from prediction
#             if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
#                 pred_indices = torch.arange(num_matches_pred, device=_device)
#             else:
#                 pred_indices = torch.randint(
#                     num_matches_pred,
#                     (num_matches_train - self.train_pad_num_gt_min,),
#                     device=_device)
#
#             # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
#             gt_pad_indices = torch.randint(
#                 len(data['spv_b_ids']),
#                 (max(num_matches_train - num_matches_pred,
#                      self.train_pad_num_gt_min),),
#                 device=_device)
#             mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero
#
#             b_ids, i_ids, j_ids, mconf = map(
#                 lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
#                                        dim=0),
#                 *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
#                      [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))
#
#         # These matches select patches that feed into fine-level network
#         coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}
#
#         # # 4. Update with matches in original image resolution
#         # scale = 8
#         # scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
#         # scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
#         # mkpts0_c = torch.stack(
#         #     [i_ids % 40, i_ids // 40],
#         #     dim=1) * scale0
#         # mkpts1_c = torch.stack(
#         #     [j_ids % 40, j_ids // 40],
#         #     dim=1) * scale1
#         #
#         # # These matches is the current prediction (for visualization)
#         # coarse_matches.update({
#         #     'gt_mask': mconf == 0,
#         #     'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
#         #     'mkpts0_c': mkpts0_c[mconf != 0],
#         #     'mkpts1_c': mkpts1_c[mconf != 0],
#         #     'mconf': mconf[mconf != 0]
#         # })
#
#         return coarse_matches
#
#
# @torch.no_grad()
# def mask_pts_at_padded_regions(grid_pt, mask):
#     """For megadepth dataset, zero-padding exists in images"""
#     mask = repeat(mask, 'n h w -> n (h w) c', c=2)
#     grid_pt[~mask.bool()] = 0
#     return grid_pt
#
#
# @torch.no_grad()
# def spvs_coarse(data, config):
#     """
#     Update:
#         data (dict): {
#             "conf_matrix_gt": [N, hw0, hw1],
#             'spv_b_ids': [M]
#             'spv_i_ids': [M]
#             'spv_j_ids': [M]
#             'spv_w_pt0_i': [N, hw0, 2], in original image resolution
#             'spv_pt1_i': [N, hw1, 2], in original image resolution
#         }
#
#     NOTE:
#         - for scannet dataset, there're 3 kinds of resolution {i, c, f}
#         - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
#     """
#     # 1. misc
#     device = data['image0'].device
#     N, _, H0, W0 = data['image0'].shape
#     _, _, H1, W1 = data['image1'].shape
#     scale = config['loftr']['resolution'][0]
#     scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
#     scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
#     h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
#
#     # 2. warp grids
#     # create kpts in meshgrid and resize them to image resolution
#     grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)  # [N, hw, 2]
#     grid_pt0_i = scale0 * grid_pt0_c
#     grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N, 1, 1)
#     grid_pt1_i = scale1 * grid_pt1_c
#
#     # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
#     if 'mask0' in data:
#         grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
#         grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])
#
#     # warp kpts bi-directionally and resize them to coarse-level resolution
#     # (no depth consistency check, since it leads to worse results experimentally)
#     # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
#     # _, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
#     # _, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
#     w_pt0_c = grid_pt0_i / scale1
#     w_pt1_c = grid_pt1_i / scale0
#
#     # 3. check if mutual nearest neighbor
#     w_pt0_c_round = w_pt0_c[:, :, :].round().long()
#     nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
#     w_pt1_c_round = w_pt1_c[:, :, :].round().long()
#     nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0
#
#     # corner case: out of boundary
#     def out_bound_mask(pt, w, h):
#         return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
#
#     nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
#     nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0
#
#     loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
#     correct_0to1 = loop_back == torch.arange(h0 * w0, device=device)[None].repeat(N, 1)
#     correct_0to1[:, 0] = False  # ignore the top-left corner
#
#     # 4. construct a gt conf_matrix
#     conf_matrix_gt = torch.zeros(N, h0 * w0, h1 * w1, device=device)
#     b_ids, i_ids = torch.where(correct_0to1 != 0)
#     j_ids = nearest_index1[b_ids, i_ids]
#
#     conf_matrix_gt[b_ids, i_ids, j_ids] = 1
#     data.update({'conf_matrix_gt': conf_matrix_gt})
#
#     # 5. save coarse matches(gt) for training fine level
#     if len(b_ids) == 0:
#         # logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
#         # this won't affect fine-level loss calculation
#         b_ids = torch.tensor([0], device=device)
#         i_ids = torch.tensor([0], device=device)
#         j_ids = torch.tensor([0], device=device)
#
#     data.update({
#         'spv_b_ids': b_ids,
#         'spv_i_ids': i_ids,
#         'spv_j_ids': j_ids
#     })
#
#     # # 6. save intermediate results (for fast fine-level computation)
#     # data.update({
#     #     'spv_w_pt0_i': w_pt0_i,
#     #     'spv_pt1_i': grid_pt1_i
#     # })
