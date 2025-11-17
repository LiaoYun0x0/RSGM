import argparse
import os
import sys
import random
import json
import numpy as np
import torch
import time
from typing import Iterable, Optional
from torch.utils.data import (DataLoader, SequentialSampler)
from tqdm import tqdm
import logging
import colorlog

import util
from models import build_model
# --- 【重要】确保您的 build_dataset 在此 ---
from datasets import build_dataset
# -----------------------------------------

from common.functions import * # 假设 eval_src_homography, cal_error_auc 在这里
from configs import dynamic_load
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 日志记录器设置 ---
LOGFORMAT = "[%(log_color)s%(levelname)s] [%(log_color)s%(asctime)s] %(log_color)s%(filename)s [line:%(log_color)s%(lineno)d] : %(log_color)s%(message)s%(reset)s"
formatter = colorlog.ColoredFormatter(LOGFORMAT)
LOG_LEVEL = logging.INFO
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger()
log.setLevel(LOG_LEVEL)
log.addHandler(stream)


# --------------------

def _transform_inv(img, mean, std):
    img = img * std + mean
    img = np.uint8(img * 255.0)
    img = img.transpose(1, 2, 0)
    return img


@torch.no_grad()
def run_evaluation(
        loader: Iterable, model: torch.nn.Module
):
    """
    运行评估循环：计算指标 (Metrics) - AUC, Runtime, Retained Rate
    """
    model.eval()

    header = 'Evaluating'

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_gpu_time_ms = 0.0

    thres = [5, 10, 20]
    dists_sa = []
    retained_rates = []

    mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std_np = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    for sample_batch in tqdm(loader, desc=header):
        images1 = sample_batch["refer"].cuda().float()
        images0 = sample_batch["query"].cuda().float()
        gt_matrix = 0
        H_gt = sample_batch['h_gt'].squeeze(0)

        starter.record()
        preds = model(images0, images1, gt_matrix)
        ender.record()
        torch.cuda.synchronize()
        total_gpu_time_ms += starter.elapsed_time(ender)

        retained_rates.append(preds.get('retained_rate', torch.tensor(0.0)).item())

        samples0_np = _transform_inv(images0.detach().cpu().numpy().squeeze(0), mean_np, std_np)
        samples1_np = _transform_inv(images1.detach().cpu().numpy().squeeze(0), mean_np, std_np)

        mkpts0 = preds.get('mkpts0', torch.empty((0, 3)).to(DEV))
        mkpts1 = preds.get('mkpts1', torch.empty((0, 2)).to(DEV))

        if mkpts0.shape[0] > 0 and mkpts1.shape[0] > 0:
            # mkpts0_np = mkpts0[:, 1:].detach().cpu().numpy()
            # mkpts1_np = mkpts1.detach().cpu().numpy()
            # H_gt_np = H_gt.detach().cpu().numpy()

            dist, d_gt = eval_src_homography(mkpts0[:, 1:], mkpts1, samples0_np, samples1_np, H_gt)
            dists_sa.append(dist)
            # dists_sa.append(dist)
        else:
            dists_sa.append(float('inf'))

    results = {}
    if len(dists_sa) > 0:
        # --- 【修复】 依据 train_SAR2RGB-base.py ---
        # 在计算 AUC 之前过滤掉 nan 值
        dists_sa_cleaned = [d for d in dists_sa if not np.isnan(d)]
        # -----------------------------------------------

        # 使用过滤后的列表进行计算
        auc_sa = cal_error_auc(dists_sa_cleaned, thresholds=thres)
        results['auc_5'] = auc_sa[0]
        results['auc_10'] = auc_sa[1]
        results['auc_20'] = auc_sa[2]

        avg_time_ms = total_gpu_time_ms / len(loader)
        results['runtime_ms'] = avg_time_ms

        avg_retained_rate = np.mean(retained_rates) * 100.0
        results['retained_rate_percent'] = avg_retained_rate
    else:
        results = {'auc_5': 0, 'auc_10': 0, 'auc_20': 0, 'runtime_ms': 0, 'retained_rate_percent': 0}

    return results


def main(args):
    util.init_distributed_mode(args)

    seed = args.seed + util.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    log.info(f'Seed used: {seed}')

    # --- 【重要】加载您的最佳模型 ---
    # !! 请将此路径修改为您训练好的 .pth 文件 !!
    best_checkpoint_path = "artifacts/avgpool/resnet101-dual_softmax_dim256-128_depth256-128/auc_4_model_nirscene1_10036.4_12.0.pth"
    # -----------------------------------

    if not os.path.exists(best_checkpoint_path):
        log.error(f"Checkpoint not found at: {best_checkpoint_path}")
        log.error("Please update 'best_checkpoint_path' in run_ablations.py to your trained model.")
        sys.exit(1)

    log.info(f"Loading checkpoint: {best_checkpoint_path}")
    state_dict = torch.load(best_checkpoint_path, map_location='cpu', weights_only=True)  # 增加 weights_only=True

    original_args = argparse.Namespace(**vars(args))

    # --- 1. 运行 Table 3 (Runtime) ---
    log.info("\n" + "=" * 40 + "\nRunning Ablation for Table 3 (Runtime)\n" + "=" * 40)

    # --- 【重要】确保这些键名与您的论文 Table 3 一致 ---
    # 并且与 model.py forward 函数中的 getattr(self.args, '...') 一致
    ablation_toggles = {
        'Full_Model': {},
        'No_CNIM(PF)': {'use_cnim': False},  # 对应 'use_cnim'
        'No_SGM': {'use_sgm': False},  # 对应 'use_sgm'
        'No_MM': {'use_mm': False},  # 对应 'use_mm'
        # 'No_SIFM(SF)': {'use_sifm': False},
        # 'No_NPASC(SC)': {'use_npasc': False},
    }

    results_table3 = {}

    for name, toggles in ablation_toggles.items():
        log.info(f"\n--- Testing Table 3 config: {name} ---")
        current_args = argparse.Namespace(**vars(original_args))

        for key, value in toggles.items():
            setattr(current_args, key, value)
            log.info(f"Setting {key} = {value}")

        current_args.eval_image_size = 512  # 评估在 512 上
        # 训练尺寸必须与加载的 .pth 文件一致！
        current_args.train_image_size = 320  # 假设模型是 320 训练的

        model: torch.nn.Module = build_model(current_args)
        model = model.to(DEV)
        model.load_state_dict(state_dict['model'], strict=False)  # strict=False 允许部分加载

        _, test_dataset = build_dataset(current_args)
        test_sampler = SequentialSampler(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=0)

        eval_stats = run_evaluation(test_loader, model)
        results_table3[name] = eval_stats
        log.info(
            f"Results for {name}: AUC@5/10/20=[{eval_stats['auc_5']:.2f}, {eval_stats['auc_10']:.2f}, {eval_stats['auc_20']:.2f}], Runtime={eval_stats['runtime_ms']:.2f} ms")

    print("\n--- Table 3 Results Summary ---")
    print(json.dumps(results_table3, indent=4))

    # --- 2. 运行 Table 6 (K, alpha, beta) ---
    log.info("\n" + "=" * 40 + "\nRunning Ablation for Table 6 (K, alpha, beta)\n" + "=" * 40)

    K_values = [3, 5, 7]
    alpha_beta_values = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]
    results_table6 = {}

    # a) 改变 K
    for k in K_values:
        log.info(f"\n--- Testing Table 6 config: K={k} ---")
        current_args = argparse.Namespace(**vars(original_args))
        current_args.eval_image_size = 512
        current_args.train_image_size = 320
        setattr(current_args, 'neighbor_k', k)
        setattr(current_args, 'fusion_alpha', 0.5)
        setattr(current_args, 'fusion_beta', 0.5)

        model = build_model(current_args).to(DEV)
        model.load_state_dict(state_dict['model'], strict=False)

        _, test_dataset = build_dataset(current_args)
        test_loader = DataLoader(test_dataset, batch_size=1, sampler=SequentialSampler(test_dataset), num_workers=0)

        eval_stats = run_evaluation(test_loader, model)
        results_table6[f'K={k}'] = eval_stats
        log.info(
            f"Results for K={k}: AUC@5/10/20=[{eval_stats['auc_5']:.2f}, {eval_stats['auc_10']:.2f}, {eval_stats['auc_20']:.2f}]")

    # b) 改变 alpha, beta
    for alpha, beta in alpha_beta_values:
        log.info(f"\n--- Testing Table 6 config: alpha={alpha}, beta={beta} ---")
        current_args = argparse.Namespace(**vars(original_args))
        current_args.eval_image_size = 512
        current_args.train_image_size = 320
        setattr(current_args, 'neighbor_k', 3)
        setattr(current_args, 'fusion_alpha', alpha)
        setattr(current_args, 'fusion_beta', beta)

        model = build_model(current_args).to(DEV)
        model.load_state_dict(state_dict['model'], strict=False)

        _, test_dataset = build_dataset(current_args)
        test_loader = DataLoader(test_dataset, batch_size=1, sampler=SequentialSampler(test_dataset), num_workers=0)

        eval_stats = run_evaluation(test_loader, model)
        results_table6[f'alpha={alpha},beta={beta}'] = eval_stats
        log.info(
            f"Results for alpha={alpha},beta={beta}: AUC@5/10/20=[{eval_stats['auc_5']:.2f}, {eval_stats['auc_10']:.2f}, {eval_stats['auc_20']:.2f}]")

    print("\n--- Table 6 Results Summary ---")
    print(json.dumps(results_table6, indent=4))

    # --- 3. 运行 Table 7 (Threshold p) ---
    log.info("\n" + "=" * 40 + "\nRunning Ablation for Table 7 (Threshold p)\n" + "=" * 40)
    p_values = [0.6, 0.7, 0.8, 0.9]
    results_table7 = {}

    for p in p_values:
        log.info(f"\n--- Testing Table 7 config: p={p} ---")
        current_args = argparse.Namespace(**vars(original_args))
        current_args.eval_image_size = 512
        current_args.train_image_size = 320
        setattr(current_args, 'adaptive_threshold_p', p)

        model = build_model(current_args).to(DEV)
        model.load_state_dict(state_dict['model'], strict=False)

        _, test_dataset = build_dataset(current_args)
        test_loader = DataLoader(test_dataset, batch_size=1, sampler=SequentialSampler(test_dataset), num_workers=0)

        eval_stats = run_evaluation(test_loader, model)
        results_table7[f'p={p}'] = eval_stats
        log.info(
            f"Results for p={p}: AUC@5={eval_stats['auc_5']:.2f}, RetainedRate={eval_stats['retained_rate_percent']:.2f}%")

    print("\n--- Table 7 Results Summary ---")
    print(json.dumps(results_table7, indent=4))

    # --- 最终保存 ---
    final_results = {
        "Table3_Runtimes": results_table3,
        "Table6_Hyperparams": results_table6,
        "Table7_Threshold_p": results_table7
    }
    save_path = os.path.join(args.save_path, "ablation_results.json")
    with open(save_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    log.info(f"\nAll ablation results saved to {save_path}")
    print('Finished Evaluation!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str,
                        default='imcnet_config')

    parser.add_argument('--train_image_size', type=int, default=320,
                        help="Image size used during training (for model build)")
    parser.add_argument('--eval_image_size', type=int, default=512,
                        help="Image size for evaluation (e.g., 512)")

    parser.add_argument('--batch_size', type=int, default=1, help="Batch size (must be 1 for eval)")
    parser.add_argument('--seed', type=int, default=700)
    parser.add_argument('--save_path', type=str, default='ablation_output')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--use_sifm', type=bool, default=None)
    parser.add_argument('--use_cnim', type=bool, default=None)
    parser.add_argument('--use_sgm', type=bool, default=None)
    parser.add_argument('--use_npasc', type=bool, default=None)
    parser.add_argument('--use_mm', type=bool, default=None)
    parser.add_argument('--neighbor_k', type=int, default=None)
    parser.add_argument('--fusion_alpha', type=float, default=None)
    parser.add_argument('--fusion_beta', type=float, default=None)
    parser.add_argument('--adaptive_threshold_p', type=float, default=None)

    # --- 【修复】 typo 'add_KNOWN_args' -> 'parse_known_args' ---
    global_cfgs, unknown = parser.parse_known_args()
    # -----------------------------------------------------

    args_from_config = dynamic_load(global_cfgs.config_name)

    cmd_args_dict = vars(global_cfgs)
    config_vars_dict = vars(args_from_config)

    config_vars_dict.update({k: v for k, v in cmd_args_dict.items() if v is not None})

    args = argparse.Namespace(**config_vars_dict)

    prm_str = 'Arguments:\n' + '\n'.join(
        ['{} {}'.format(k.upper(), v) for k, v in vars(args).items()]
    )
    print(prm_str + '\n')
    print('==' * 40 + '\n')

    main(args)