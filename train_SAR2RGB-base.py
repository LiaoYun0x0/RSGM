import argparse
import json
import logging
import os
import random
from typing import Iterable

import colorlog
import cv2
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (DataLoader, BatchSampler, RandomSampler,
                              SequentialSampler, DistributedSampler)
from tqdm import tqdm

import util
from common.functions import *
from common.logger import Logger, MetricLogger, SmoothedValue
from configs import dynamic_load
from datasets import build_dataset
from loss import build_criterion
from models import build_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

colorlog.basicConfig(format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', filename='myapp.log',
                     filemode='w', datefmt='%a, %d %b %Y %H:%M:%S', )

LOGFORMAT = "[%(log_color)s%(levelname)s] [%(log_color)s%(asctime)s] %(log_color)s%(filename)s [line:%(log_color)s%(lineno)d] : %(log_color)s%(message)s%(reset)s"
formatter = colorlog.ColoredFormatter(LOGFORMAT)
LOG_LEVEL = logging.NOTSET
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(LOG_LEVEL)
log.addHandler(stream)


# (在 train_SAR2RGB-base.py 中)
# (确保你导入了: tqdm, util, cv2, np, torch, logging.log)
# (确保你从 common.functions 导入了: draw_match_nir, eval_src_mma, eval_src_homography, cal_error_auc)

@torch.no_grad()
def test(
        loader: Iterable, model: torch.nn.Module, criterion: torch.nn.Module, print_freq=100., tb_logger=None
):
    model.eval()

    def _transform_inv(img, mean, std):
        # (img 必须是 (C, H, W) 格式)
        img = img * std + mean
        img = np.uint8(img * 255.0)
        img = img.transpose(1, 2, 0)
        return img

    logger = MetricLogger(delimiter=' ')
    header = 'Test'
    scores = 0
    i_err = {thr: 0 for thr in np.arange(1, 11)}
    thres = [5, 10, 20]
    nums = 0
    dists_sa = []
    dists_gt = []
    IM_POS = 0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    for sample_batch in tqdm(loader):
        # --- 1. 获取批处理数据 (B = Batch Size) ---
        images1_b = sample_batch["refer"].cuda().float()
        images0_b = sample_batch["query"].cuda().float()
        gt_matrix_b = sample_batch['gt_matrix'].cuda().float()
        h_gt_b = sample_batch['h_gt'].cuda().float()

        # --- 2. 运行模型 (在整个批处理上) ---
        preds = model(images0_b, images1_b, gt_matrix_b)

        targets_b = {
            'gt_matrix': gt_matrix_b,
            'h_gt': h_gt_b
        }

        # --- 3. 计算批处理损失 (在循环外) ---
        loss_dict = criterion(preds, targets_b)
        loss_dict_reduced = util.reduce_dict(loss_dict)
        loss_dict_reduced_item = {
            k: v.item() for k, v in loss_dict_reduced.items()
        }
        logger.update(**loss_dict_reduced_item)

        # --- 4. 【修复】在批处理内部循环以进行评估 ---

        preds_mkpts0_b = preds['mkpts0']
        preds_mkpts1_b = preds['mkpts1']

        batch_size = images0_b.shape[0]
        for b_idx in range(batch_size):
            # --- 5. 提取单个样本 ---
            img0_tensor = images0_b[b_idx]
            img1_tensor = images1_b[b_idx]
            h_gt_single = h_gt_b[b_idx].cpu()

            # --- 6. 筛选此样本的匹配点 ---
            mask_b = (preds_mkpts0_b[:, 0] == b_idx)
            mkpts0_single = preds_mkpts0_b[mask_b][:, 1:]
            mkpts1_single = preds_mkpts1_b[mask_b]

            # --- 7. 安全地调用 _transform_inv 和评估 ---
            samples0 = _transform_inv(img0_tensor.detach().cpu().numpy(), mean, std)
            samples1 = _transform_inv(img1_tensor.detach().cpu().numpy(), mean, std)

            out2, _ = draw_match_nir(mkpts0_single, mkpts1_single, samples0, samples1, 0, 0)
            cv2.imwrite(f"./result/{IM_POS}.jpg", out2)
            IM_POS += 1

            i_err, num = eval_src_mma(mkpts0_single, mkpts1_single, samples0, samples1, i_err, h_gt_single)

            # --- 【修复】: 现在 eval_src_homography 返回两个值 ---
            dist, d_gt = eval_src_homography(mkpts0_single, mkpts1_single, samples0, samples1, h_gt_single)

            dists_sa.append(dist)  # dist 可能是 np.nan
            dists_gt.append(d_gt)  # d_gt 可能是 np.nan (np.float64 类型)

            nums += 1

    # --- 8. 循环结束，计算最终指标 ---

    # --- 【修复】: 在计算 AUC 之前过滤 nan ---
    dists_sa_cleaned = [d for d in dists_sa if not np.isnan(d)]
    dists_gt_cleaned = [d for d in dists_gt if not np.isnan(d)]  # d_gt 已经是 float 或 nan

    auc_sa = cal_error_auc(dists_sa_cleaned, thresholds=thres)
    auc_dt = cal_error_auc(dists_gt_cleaned, thresholds=thres)

    for thr in i_err:
        if nums > 0:
            i_err[thr] = i_err[thr] / nums

    print(f"\nMMA:{i_err}")
    print(f"dists_sa:{dists_sa}")
    print(f"dists_gt (raw):{dists_gt}")
    print(f"auc_sa:{auc_sa}")
    print(f"auc_dt:{auc_dt}")

    log.info(f'Average  test stats: {logger}')
    return {k: meter.global_avg for k, meter in logger.meters.items()}, auc_sa

def train(
        epoch: int, loader: Iterable, model: torch.nn.Module,
        criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
        max_norm=0., print_freq=1000., tb_logger=None
):
    model.train()
    criterion.train()

    logger = MetricLogger(delimiter=' ')
    logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = f'Epoch: [{epoch}]'

    for sample_batch in logger.log_every(tqdm(loader), print_freq, header):
        # print(sample_batch)
        images1 = sample_batch["refer"].cuda().float()
        images0 = sample_batch["query"].cuda().float()
        gt_matrix = sample_batch['gt_matrix'].cuda().float()
        h_gt = sample_batch['h_gt'].cuda().float()
        preds = model(images0, images1, gt_matrix)
        # for name, parms in model.named_parameters():
        #    print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
        #    if parms.data is not None and parms.grad is not None:
        #        print('--weight',  torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
        targets = {
            'gt_matrix': gt_matrix,
            'h_gt': h_gt
        }

        loss_dict = criterion(preds, targets)
        loss = loss_dict['losses']
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm
            )
        optimizer.step()

        loss_dict_reduced = util.reduce_dict(loss_dict)
        loss_dict_reduced_item = {
            k: v.item() for k, v in loss_dict_reduced.items()
        }

        logger.update(**loss_dict_reduced_item)
        logger.update(lr=optimizer.param_groups[0]['lr'])
        if tb_logger is not None:
            if util.is_main_process():
                tb_logger.add_scalers(loss_dict_reduced, prefix='train')

    logger.synchronize_between_processes()
    log.info(f'Average stats:{logger}')
    return {k: meter.global_avg for k, meter in logger.meters.items()}


def main(args):
    util.init_distributed_mode(args)

    seed = args.seed + util.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Seed used:', seed)

    model: torch.nn.Module = build_model(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # model = torch.nn.DataParallel(model, device_ids=[0,1])
    print('Trainable parameters:', n_params)
    model = model.to(DEV)

    criterion = build_criterion(args)
    criterion = criterion.to(DEV)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids={args.gpu})
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=args.lr, weight_decay=args.weight_decay
    )
    # optimizer = torch.optim.Adam(model_without_ddp.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=1e-8)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
    train_dataset, test_dataset = build_dataset(args)
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        # train_sampler = SequentialSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)
    batch_train_sampler = BatchSampler(
        train_sampler, args.batch_size, drop_last=True
    )

    dataloader_kwargs = {
        # 'collate_fn': train_dataset.collate_fn,
        'pin_memory': True,
        'num_workers': 8,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_train_sampler,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        drop_last=True,
        **dataloader_kwargs
    )
    # sample = train_loader.dataset[0]  # 或者 dataset[0]

    # query = sample["query"].numpy().transpose(1, 2, 0)  # CHW -> HWC
    # refer = sample["refer"].numpy().transpose(1, 2, 0)
    # H_gt = sample["h_gt"].numpy()
    #
    # # 将 query warp 到 refer 的坐标系
    # warp_query = cv2.warpPerspective(query, H_gt, (query.shape[1], query.shape[0]))

    # 保存查看
    # cv2.imwrite("query_warped_to_refer.jpg", (warp_query * 255).astype(np.uint8))
    # cv2.imwrite("refer_img.jpg", (refer * 255).astype(np.uint8))
    # print(H_gt)
    if args.load is not None:
        state_dict = torch.load(args.load, map_location='cpu')
        model_without_ddp.load_state_dict(state_dict['model'])

    save_name = f'{args.backbone_name}-{args.matching_name}'
    save_name += f'_dim{args.d_coarse_model}-{args.d_fine_model}'
    save_name += f'_depth{args.d_coarse_model}-{args.d_fine_model}'

    save_path = os.path.join(args.save_path, save_name)
    os.makedirs(save_path, exist_ok=True)
    if util.is_main_process():
        tensorboard_logger = Logger(save_path)
    else:
        tensorboard_logger = None

    print('Start Training...')
    best_loss = 200000
    best_loss_test = 200000
    best_auc = 0
    best_epoch = 0
    for epoch in range(args.train_epoch):
        epoch = epoch
        print("\n" + "<<" * 18 + "=" * 50 + f"epoche:{epoch}" + "=" * 50 + ">>" * 18 + "\n")
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(
            epoch,
            train_loader,
            model,
            criterion,
            optimizer,
            max_norm=args.clip_max_norm,
            print_freq=args.log_interval,
            tb_logger=tensorboard_logger
        )
        scheduler.step()

        if epoch % args.save_interval == 0 or epoch == args.n_epoch - 1:
            if False:
                torch.save({
                    'model': model_without_ddp.state_dict()
                }, f'{save_path}/model-epoch{epoch}.pth')
        test_stats, auc = test(
            test_loader,
            model,
            criterion,
        )
        log_stats = {
            'epoch': epoch,
            'n_params': n_params,
            'data_name': args.data_name,
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }
        if log_stats['train_losses'] < best_loss:
            best_loss = log_stats['train_losses']
            coarse_loss = log_stats['train_coarse_loss']
            fine_loss = log_stats['train_fine_loss']
            torch.save({'model': model_without_ddp.state_dict()},
                       f'{save_path}/train_{epoch}_model_nirscene1_train_{best_loss:.3f}_{coarse_loss:.3f}_{fine_loss:.3f}.pth')
        if log_stats['test_losses'] < best_loss_test:
            best_loss_test = log_stats['test_losses']
            coarse_loss = log_stats['test_coarse_loss']
            fine_loss = log_stats['test_fine_loss']
            torch.save({'model': model_without_ddp.state_dict()},
                       f'{save_path}/test_{epoch}_model_nirscene1_test_{best_loss_test:.3f}_{coarse_loss:.3f}_{fine_loss:.3f}.pth')
        if auc[2] > best_auc:
            this_best_loss = log_stats['test_losses']
            this_fine_loss = log_stats['test_fine_loss']
            best_auc = auc[2]
            best_epoch = epoch
            torch.save({'model': model_without_ddp.state_dict()},
                       f'{save_path}/auc_{epoch}_model_nirscene1_{this_best_loss:.1f}_{this_fine_loss:.1f}.pth')
        print(f"This epoch auc:{auc},best epoch:{best_epoch},best epoch auc:{best_auc}")
        with open(f'{save_path}/train_nirscene1.log', 'a') as f:
            f.write(json.dumps(log_stats) + '\n')
    print('Finished!')


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
    print('==' * 40 + '\n')

    main(args)
