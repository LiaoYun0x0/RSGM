import argparse
import os
import sys
import random
import json
import numpy as np
import torch
import time

from typing import Iterable, Optional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (DataLoader, BatchSampler, RandomSampler,
                              SequentialSampler, DistributedSampler)
import util
from models import build_model 
from datasets import build_dataset
from loss import build_criterion 
from common.error import NoGradientError
from common.logger import Logger, MetricLogger, SmoothedValue
from common.functions import *
from common.nest import NestedTensor
from configs import dynamic_load
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def test(
    loader: Iterable, model: torch.nn.Module, print_freq=100., tb_logger=None
):
    model.eval()
    def _transform_inv(img,mean,std):
        img = img * std + mean
        img  = np.uint8(img * 255.0)
        img = img.transpose(1,2,0)
        return img


    logger = MetricLogger(delimiter=' ')
    header = 'Test'
    scores = 0
    i_err = {thr: 0 for thr in np.arange(1,11)}
    thres = [1,3,5,10]
    nums = 0
    dists_sa = []
    IM_POS = 0


    for sample_batch in logger.log_every(loader, print_freq, header):
        scores+=1
        images1 = sample_batch["refer"].cuda().float()
        images0 = sample_batch["query"].cuda().float() #[1,3,512,512]
        gt_matrix=0
        x,y=int(0), int(0)

        preds = model(images0, images1, gt_matrix)
        mean = np.array([0.485, 0.456, 0.406],dtype=np.float32).reshape(3,1,1)
        std = np.array([0.229, 0.224, 0.225],dtype=np.float32).reshape(3,1,1)

        samples0 = _transform_inv(images0.detach().cpu().numpy().squeeze(), mean, std)
        samples1 = _transform_inv(images1.detach().cpu().numpy().squeeze(), mean, std)
        samples1 = cv2.cvtColor(samples1, cv2.COLOR_BGR2RGB)


        out2 = draw_match_se(preds['mkpts0'][:, 1:], preds['mkpts1'], samples0, samples1, x, y)
        #out2 = draw_match_kpts(preds['mkpts0'][:, 1:], preds['mkpts1'], samples0, samples1, x, y)
            #out3, real =  rocket_score(preds['mkpts0'][:, 1:], preds['mkpts1'], samples0, samples1, x, y)

        #match_prec = rocket_match_num(preds['mkpts0'][:, 1:], preds['mkpts1'], samples0, samples1, x, y, match_prec)
        #f.write(' '.join([sample_batch['sar'][0], sample_batch['opt'][0], str(real[0]), str(real[1]), '\n']))
        #cv2.imshow("coarse_pred", out1)
        cv2.imwrite(f"/four_disk/wxn_work/Train_New_Model/test_window_topk_nir/test_nir_avgpool/{IM_POS}.jpg", out2)
        # cv2.imwrite(f"test_nir/loftr_nirscenens1_femit/{IM_POS}.jpg", out2)
        IM_POS += 1
        #cv2.imwrite(f"loftr_nirscenens1_kpts/{time.time()}.jpg", out2)
        i_err, num = eval_src_mma(preds['mkpts0'][:,1: ], preds['mkpts1'], samples0, samples1, i_err)
        dist = eval_src_homography(preds['mkpts0'][:,1: ], preds['mkpts1'], samples0, samples1)
        dists_sa.append(dist)
        nums += 1
    correct_sa = np.mean(
            [[float(dist <= t) for t in thres] for dist in dists_sa], axis=0)
    auc_sa = cal_error_auc(dists_sa, thresholds=thres)
    for thr in i_err:
        i_err[thr] = i_err[thr] / nums
    return i_err, auc_sa



def main(args):
    util.init_distributed_mode(args)

    seed = args.seed + util.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Seed used:', seed)

    model: torch.nn.Module = build_model(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters:', n_params)
    model = model.to(DEV)
    train_dataset, test_dataset = build_dataset(args)
    test_sampler = SequentialSampler(test_dataset)


    dataloader_kwargs = {
        #'collate_fn': train_dataset.collate_fn,
        'pin_memory': False,
        'num_workers': 0,
    }

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        drop_last=True,
        **dataloader_kwargs
    )
    res = {}
    model_names = os.listdir("/four_disk/wxn_work/Train_New_Model/train_window_topk_nir/artifacts/avgpool/resnet101-dual_softmax_dim256-128_depth256-128/")
    # model_names = os.listdir("/home/ly/Documents/zkj/Femip/loftr_nir/artifacts/resnet101-dual_softmax_dim256-128_depth256-128/")
    for model_name in [x for x in model_names if 'model_nirscene1' in x]: #179_model_nirscene1_1213.0_25.5
        state_dict = torch.load(f"/four_disk/wxn_work/Train_New_Model/train_window_topk_nir/artifacts/avgpool/resnet101-dual_softmax_dim256-128_depth256-128/{model_name}", map_location='cpu')
        model.load_state_dict(state_dict['model'],strict=False)

        save_name = f'{args.backbone_name}-{args.matching_name}'
        save_name += f'_dim{args.d_coarse_model}-{args.d_fine_model}'
        save_name += f'_depth{args.d_coarse_model}-{args.d_fine_model}'

        save_path = os.path.join(args.save_path, save_name)
        os.makedirs(save_path, exist_ok=True)
        if util.is_main_process():
            tensorboard_logger = Logger(save_path)
        else:
            tensorboard_logger = None

        print(f'Start Testing model {model_name}    ...')

        test_stats = test(
            test_loader,
            model,
        )
        print(test_stats)
        res[model_name] = {'err':test_stats[0], 'auc': test_stats[1]}
    print(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str,
                        default='imcnet_config')
    global_cfgs = parser.parse_args()

    args = dynamic_load(global_cfgs.config_name)
    prm_str = 'Arguments:\n' + '\n'.join(
        ['{} {}'.format(k.upper(), v) for k, v in vars(args).items()]
    )
    print(prm_str + '\n')
    print('=='*40 + '\n')

    main(args)
