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
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
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
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
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

class TopicFormer(nn.Module):

    def __init__(self, config):
        super(TopicFormer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names_t']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])

        # if config['n_samples'] > 0:
        self.feat_aug = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(2 * config['n_topic_transformers'])])
        self.n_iter_topic_transformer = config['n_topic_transformers']

        self.seed_tokens = nn.Parameter(torch.randn(config['n_topics'], config['d_model']))
        self.register_parameter('seed_tokens', self.seed_tokens)
        self.topic_drop = nn.Dropout1d(p=0.1)
        self.n_samples = config['n_samples']
        self.avgpool = nn.AdaptiveAvgPool1d(160)
        self.avgpool_1 = nn.AdaptiveAvgPool1d(96)
        self.norm_feat = nn.LayerNorm(self.d_model)
        # self.fea_down_t0 = conv1x1(config['d_model'] + config['d_model_fusion'] * 2, config['d_model_fusion'] * 2)
        # self.fea_down_t1 = conv1x1(config['d_model_fusion'] * 2, config['d_model_fusion'])
        # self.fea_down_t2 = conv1x1(config['d_model_fusion'], config['d_model'])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sample_topic(self, prob_topics, topics, L):
        prob_topics0, prob_topics1 = prob_topics[:, :L], prob_topics[:, L:]
        topics0, topics1 = topics[:, :L], topics[:, L:]

        theta0 = F.normalize(prob_topics0.sum(dim=1), p=1, dim=-1)  # [N, K]
        theta1 = F.normalize(prob_topics1.sum(dim=1), p=1, dim=-1)
        theta = F.normalize(theta0 * theta1, p=1, dim=-1)
        if self.n_samples == 0:
            return None
        if self.training:
            sampled_inds = torch.multinomial(theta, self.n_samples)
            sampled_values = torch.gather(theta, dim=-1, index=sampled_inds)
        else:
            sampled_values, sampled_inds = torch.topk(theta, self.n_samples, dim=-1)
        sampled_topics0 = torch.gather(topics0, dim=-1, index=sampled_inds.unsqueeze(1).repeat(1, topics0.shape[1], 1))
        sampled_topics1 = torch.gather(topics1, dim=-1, index=sampled_inds.unsqueeze(1).repeat(1, topics1.shape[1], 1))
        return sampled_topics0, sampled_topics1

    def reduce_feat(self, feat, topick, N, C):
        len_topic = topick.sum(dim=-1).int()
        max_len = len_topic.max().item()
        selected_ids = topick.bool()
        resized_feat = torch.zeros((N, max_len, C), dtype=torch.float, device=feat.device)
        new_mask = torch.zeros_like(resized_feat[..., 0]).bool()
        for i in range(N):
            new_mask[i, :len_topic[i]] = True
        resized_feat[new_mask, :] = feat[selected_ids, :]
        return resized_feat, new_mask, selected_ids

    def forward(self, feat0, feat1, topic0, topic1, mask0=None, mask1=None):

        assert self.d_model == feat0.shape[2], "the feature number of src and transformer must be equal"
        N, L, S, C, K = feat0.shape[0], feat0.shape[1], feat1.shape[1], feat0.shape[2], self.config['n_topics']
        seeds = self.seed_tokens.unsqueeze(0).repeat(N, 1, 1)
        seeds = self.topic_drop(seeds)
        topic0 = torch.transpose(topic0, -1, -2)
        topic1 = torch.transpose(topic1, -1, -2)

        top_matrix = torch.einsum("nmd,nld->nml", topic0, topic1) // C ** .5
        top_matrix = F.softmax(top_matrix, 1) * F.softmax(top_matrix, 2)
        top_matrix_idx = torch.argmax(top_matrix, -1)
        topic1 = topic1[:, top_matrix_idx[-1]]
        topic = torch.concat((topic0, topic1), -1)
        topic = self.avgpool_1(topic)
        # print(topic.shape)
        # print(seeds.shape)

        # print(seeds.shape)
        # seeds = self.fea_down_t0(torch.unsqueeze(seeds, 3))
        # seeds = self.fea_down_t1(seeds)
        # seeds = self.fea_down_t2(seeds)
        # seeds = torch.squeeze(seeds, 3)
        # seeds = torch.transpose(seeds, -1, -2)
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

        seeds = self.avgpool(seeds)
        seeds = torch.concat((seeds, topic), -1)
        seeds = self.norm_feat(seeds)
        dmatrix = torch.einsum("nmd,nkd->nmk", feat, seeds) / C ** .5
        prob_topics = F.softmax(dmatrix, dim=-1)

        feat_topics = torch.zeros_like(dmatrix).scatter_(-1, torch.argmax(dmatrix, dim=-1, keepdim=True), 1.0)

        if mask is not None:
            feat_topics = feat_topics * mask.unsqueeze(-1)
            prob_topics = prob_topics * mask.unsqueeze(-1)

        sampled_topics = self.sample_topic(prob_topics.detach(), feat_topics, L)
        if sampled_topics is not None:
            updated_feat0, updated_feat1 = torch.zeros_like(feat0), torch.zeros_like(feat1)
            s_topics0, s_topics1 = sampled_topics
            for k in range(s_topics0.shape[-1]):
                topick0, topick1 = s_topics0[..., k], s_topics1[..., k]  # [N, L+S]
                if (topick0.sum() > 0) and (topick1.sum() > 0):
                    new_feat0, new_mask0, selected_ids0 = self.reduce_feat(feat0, topick0, N, C)
                    new_feat1, new_mask1, selected_ids1 = self.reduce_feat(feat1, topick1, N, C)
                    for idt in range(self.n_iter_topic_transformer):
                        new_feat0 = self.feat_aug[idt * 2](new_feat0, new_feat0, new_mask0, new_mask0)
                        new_feat1 = self.feat_aug[idt * 2](new_feat1, new_feat1, new_mask1, new_mask1)
                        new_feat0 = self.feat_aug[idt * 2 + 1](new_feat0, new_feat1, new_mask0, new_mask1)
                        new_feat1 = self.feat_aug[idt * 2 + 1](new_feat1, new_feat0, new_mask1, new_mask0)
                    updated_feat0[selected_ids0, :] = new_feat0[new_mask0, :]
                    updated_feat1[selected_ids1, :] = new_feat1[new_mask1, :]

            feat0 = (1 - s_topics0.sum(dim=-1, keepdim=True)) * feat0 + updated_feat0
            feat1 = (1 - s_topics1.sum(dim=-1, keepdim=True)) * feat1 + updated_feat1
        else:
            for idt in range(self.n_iter_topic_transformer * 2):
                feat0 = self.feat_aug[idt](feat0, seeds, mask0, None)
                feat1 = self.feat_aug[idt](feat1, seeds, mask1, None)

        # if self.training:
        # topic_matrix = torch.einsum("nlk,nsk->nls", prob_topics[:, :L], prob_topics[:, L:])
        # topic_matrix = torch.einsum("nlk,nsk->nls", prob_topics[:, :L], prob_topics[:, L:]) / C ** .5
        # print(topic_matrix)
        topic_matrix_img0 = dmatrix[:, :L]
        Nt, Ct, Ht, Wt = topic_matrix_img0.shape[0], topic_matrix_img0.shape[2], int(math.sqrt(
            topic_matrix_img0.shape[1])), int(math.sqrt(topic_matrix_img0.shape[1]))

        topic_matrix_img0 = torch.reshape(topic_matrix_img0, (Nt, Ct, Ht, Wt))

        topic_matrix_img0_unfold = F.unfold(topic_matrix_img0, kernel_size=(3, 3), stride=1, padding=1)
        topic_matrix_img0_unfold = rearrange(topic_matrix_img0_unfold, 'n (c ww) l -> n l ww c',
                                             ww=3 ** 2)  # 2*144*9*256
        topic_matrix_img0_unfold = torch.mean(torch.transpose(topic_matrix_img0_unfold, -1, -2), -1)
        # topic_matrix_img0_unfold_mask = topic_matrix_img0_unfold > 0.5
        # topic_matrix_img0_unfold_mask = topic_matrix_img0_unfold_mask.byte()
        # topic_matrix_img0_unfold_mask = torch.reshape(topic_matrix_img0_unfold_mask, (Nt, Ht * Wt, Ct))
        topic_matrix_img0 = torch.reshape(topic_matrix_img0, (Nt, Ht * Wt, Ct))

        topic_matrix_img0 = topic_matrix_img0 * topic_matrix_img0_unfold

        topic_matrix_img1 = dmatrix[:, L:]

        topic_matrix_img1 = torch.reshape(topic_matrix_img1, (Nt, Ct, Ht, Wt))

        topic_matrix_img1_unfold = F.unfold(topic_matrix_img1, kernel_size=(3, 3), stride=1, padding=1)
        topic_matrix_img1_unfold = rearrange(topic_matrix_img1_unfold, 'n (c ww) l -> n l ww c',
                                             ww=3 ** 2)  # 2*144*9*256
        topic_matrix_img1_unfold = torch.mean(torch.transpose(topic_matrix_img1_unfold, -1, -2), -1)
        # topic_matrix_img1_unfold_mask = topic_matrix_img1_unfold > 0.5
        # topic_matrix_img1_unfold_mask = topic_matrix_img1_unfold_mask.byte()
        topic_matrix_img1 = torch.reshape(topic_matrix_img1, (Nt, Ht * Wt, Ct))
        # topic_matrix_img1_unfold_mask = torch.reshape(topic_matrix_img1_unfold_mask, (Nt, Ht * Wt, Ct))
        topic_matrix_img1 = topic_matrix_img1 * topic_matrix_img1_unfold
        topic_matrix = torch.einsum("nlk,nsk->nls", prob_topics[:, :L], prob_topics[:, L:])
        # print(topic_matrix)
        topic_matrix_match = {"img0": topic_matrix_img0, "img1": topic_matrix_img1}

        # else:
        #     topic_matrix = {"img0": feat_topics[:, :L], "img1": feat_topics[:, L:]}
        #     topic_matrix_match = {"img0": feat_topics[:, :L], "img1": feat_topics[:, L:]}

        return feat0, feat1, topic_matrix_match, topic_matrix_match


class FineNetwork(nn.Module):

    def __init__(self, config, add_detector=True):
        super(FineNetwork, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.n_mlp_mixer_blocks = config["n_mlp_mixer_blocks"]
        self.encoder_layers = nn.ModuleList([MLPMixerEncoderLayer(config["n_feats"] * 2, self.d_model)
                                             for _ in range(self.n_mlp_mixer_blocks)])
        self.detector = None
        if add_detector:
            self.detector = nn.Sequential(MLPMixerEncoderLayer(config["n_feats"], self.d_model),
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

        feat = torch.cat((feat0, feat1), dim=1)
        for idx in range(self.n_mlp_mixer_blocks):
            feat = self.encoder_layers[idx](feat)
        feat0, feat1 = feat[:, :feat0.shape[1]], feat[:, feat0.shape[1]:]
        score_map0 = None
        if self.detector is not None:
            score_map0 = self.detector(feat0).squeeze(-1)

        return feat0, feat1, score_map0


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
        # self.down_dim0 = conv1x1(self.d_model_fusion * 2, self.d_model * 3 // 2)
        # self.down_dim1 = conv1x1(self.d_model_fusion * 3 // 2, self.d_model)
        # self.down_dim = conv1x1(self.d_model_fusion, self.d_model - 96)
        self.head = 4
        self.thr = config['thr']
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_s0, feat_s1, data=None):
        H = W = int(math.sqrt(feat_s0.shape[1]))
        N, C, K = feat_s0.shape[0], feat_s0.shape[2], self.config['n_topics']

        assert self.d_model_fusion == C, "the feature number of src and transformer must be equal"

        feat_s0 = torch.reshape(feat_s0, (N, H * W, C))
        feat_s1 = torch.reshape(feat_s1, (N, H * W, C))

        conf_matrix = torch.einsum("nlc,nsc->nls", feat_s0, feat_s1) / self.d_model ** .5  # (C * temperature)

        feat_s0 = torch.reshape(feat_s0, (N, H * W, C))
        feat_s1 = torch.reshape(feat_s1, (N, H * W, C))
        conf_matrix = conf_matrix

        data["conf_matrix"] = conf_matrix
        conf_matrix = F.softmax(conf_matrix, 1) * F.softmax(conf_matrix, 2)
        conf_mask = conf_matrix > self.thr
        conf_mask = conf_mask * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                    * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        conf_mask = conf_mask.float()  # 2*144*144

        feat_s0 = torch.reshape(feat_s0, (N, C, H, W))
        feat_s1 = torch.reshape(feat_s1, (N, C, H, W))

        feat_s0_unfold = F.unfold(feat_s0, kernel_size=(3, 3), stride=1, padding=1)
        feat_s0_unfold = rearrange(feat_s0_unfold, 'n (c ww) l -> n l ww c', ww=3 ** 2)  # 2*144*9*256

        feat_s0 = torch.reshape(feat_s0, (N, H * W, C))
        feat_s0_sem = torch.einsum("nlc,nlwc->nlw", feat_s0, feat_s0_unfold) / feat_s0_unfold.shape[-1]  # 2*144*9

        feat_s1_unfold = F.unfold(feat_s1, kernel_size=(3, 3), stride=1, padding=1)
        feat_s1_unfold = rearrange(feat_s1_unfold, 'n (c ww) l -> n l ww c', ww=3 ** 2)
        feat_s1 = torch.reshape(feat_s1, (N, H * W, C))
        feat_s1_sem = torch.einsum("nlc,nlwc->nlw", feat_s1, feat_s1_unfold) / feat_s1_unfold.shape[-1]  # 2*144*9

        feat_s0_fea = torch.einsum("nlwc,nlw->nlc", feat_s0_unfold, feat_s0_sem)  # 2*144*256
        feat_s1_fea = torch.einsum("nlwc,nlw->nlc", feat_s1_unfold, feat_s1_sem)  # 2*144*256
        feat_s0_sem_fea = torch.einsum("nlc,ncd->nld", conf_mask, feat_s1_fea) / conf_mask.shape[1]

        conf_mask_trans = torch.transpose(conf_mask, -2, -1)
        feat_s1_sem_fea = torch.einsum("nlc,ncd->nld", conf_mask_trans, feat_s0_fea) / conf_mask_trans.shape[1]

        conf_matrix_fusion = torch.einsum("nlc,nsc->nls", feat_s0_sem_fea,
                                          feat_s1_sem_fea) / self.d_model ** .5  # (C * temperature)
        data['conf_matrix_fusion'] = conf_matrix_fusion
        feat_s0 = self.avgpool(feat_s0)
        feat_s0_sem_fea = self.avgpool_1(feat_s0_sem_fea)
        feat_s1 = self.avgpool(feat_s1)
        feat_s1_sem_fea = self.avgpool_1(feat_s1_sem_fea)
        feat_s0 = torch.concat((feat_s0, feat_s0_sem_fea), -1)
        feat_s1 = torch.concat((feat_s1, feat_s1_sem_fea), -1)
        feat_s = torch.concat((feat_s0, feat_s1), 0)
        feat_s = self.norm_feat(feat_s)
        # feat_s = torch.reshape(feat_s, (N * 2, C * 2, H, W))
        # feat_s = self.down_dim0(feat_s)
        # feat_s = self.down_dim1(feat_s)
        # feat_s = torch.reshape(feat_s, (N * 2, H * W, self.d_model))

        return feat_s[:feat_s0.shape[0]], feat_s[feat_s0.shape[0]:]


class Relator_Fusion(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(Relator_Fusion, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.d_model_relator = config['d_model']
        self.up_dim_ini = config['relator_dim_ini']
        self.avgpool = nn.AdaptiveAvgPool1d(160)
        self.avgpool_1 = nn.AdaptiveAvgPool1d(96)
        # self.down_dim0 = conv1x1(self.d_model * 2, self.d_model * 3 // 2)
        # self.down_dim1 = conv1x1(self.d_model * 3 // 2, self.d_model)
        # self.layer_names = config['layer_names_relator']
        # encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        # self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self.norm_feat = nn.LayerNorm(self.d_model)
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
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            pos1,pos2:  [N, C, H, W]
        Outputs:
            feat0: [N,-1,C]
            feat1: [N,-1,C]
            flow_list: [L,N,H,W,4]*1(2)
        """
        # print(feat0.shape)
        # print(feat1.shape)
        relator = torch.einsum("nlc,nsc->nls", feat0, feat1) / self.d_model ** .5  # B,L,S
        relator = torch.unsqueeze(relator, 1)
        conf_matrix_fusion = data['conf_matrix_fusion']
        conf_matrix_fusion = torch.unsqueeze(conf_matrix_fusion, 1)
        conf_matrix = data['conf_matrix']
        conf_matrix = torch.unsqueeze(conf_matrix, 1)
        relators = torch.concat((conf_matrix, relator, conf_matrix_fusion), 1)
        data["conf_matrix_relator"] = torch.squeeze(torch.mean(relators, 1), 1)
        relator_trans = torch.transpose(relators, -1, -2)
        relators = torch.concat((relators, relator_trans), 0)
        relators = torch.reshape(relators,
                                 (relators.shape[0], relators.shape[2], relators.shape[3] * relators.shape[1]))
        # print(relators.shape)
        # relators = self.relators_cnn(relators)
        relators = self.avgpool_1(relators)
        # print(relators.shape)

        # relators = torch.transpose(relators, -1, -2)
        feat = torch.concat((feat0, feat1), 0)
        # print(feat.shape)
        feat = self.avgpool(feat)
        # for layer, name in zip(self.layers, self.layer_names):
        #     if name == 'feat':
        #         feat = layer(feat, relators, None)
        feat = torch.concat((feat, relators), -1)
        # feat = self.down_dim0(torch.reshape(feat, (feat.shape[0], feat.shape[2], int(math.sqrt(
        #     feat.shape[1])), int(math.sqrt(feat.shape[1])))))
        # feat = self.down_dim1(feat)
        # feat = self.avgpool(feat)
        # feat = torch.reshape(feat, (feat.shape[0], feat.shape[2] * feat.shape[3], feat.shape[1]))
        feat = self.norm_feat(feat)

        feat0 = feat[:feat0.shape[0]]
        feat1 = feat[feat0.shape[0]:]

        return feat0, feat1
