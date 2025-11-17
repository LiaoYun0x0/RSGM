import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.einops import rearrange, repeat
import cv2
import pydegensac
import random


def cal_reproj_dists(p1s, p2s, homography):
    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s.astype(np.float32) - p2s_proj.astype(np.float32)) ** 2, axis=1))
    return dist


def cal_error_auc(errors, thresholds):
    if len(errors) == 0:
        return np.zeros(len(thresholds))
    N = len(errors)
    errors = np.append([0.], np.sort(errors))
    recalls = np.arange(N + 1) / N
    aucs = []
    for thres in thresholds:
        last_index = np.searchsorted(errors, thres)
        rcs_ = np.append(recalls[:last_index], recalls[last_index - 1])
        errs_ = np.append(errors[:last_index], thres)
        aucs.append(np.trapz(rcs_, x=errs_) / thres)
    return np.array(aucs)


def make_grid(cols, rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    xs = np.tile(xs[np.newaxis, :], (rows, 1))
    ys = np.tile(ys[:, np.newaxis], (1, cols))
    grid = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1).copy()
    return grid


def _transform_inv(img, mean, std):
    img = img * std + mean
    img = np.uint8(img * 255.0)
    img = img.transpose(1, 2, 0)
    return img


def get_mkpts(matches, query, refer, patch_size=8):
    query_pts = []
    refer_pts = []
    wq = query.shape[2]
    wr = refer.shape[2]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in matches:
        x0 = patch_size / 2 + (pt[1] % qcols) * patch_size
        y0 = patch_size / 2 + (pt[1] // qcols) * patch_size
        x1 = patch_size / 2 + (pt[2] % rcols) * patch_size
        y1 = patch_size / 2 + (pt[2] // rcols) * patch_size
        query_pts.append(torch.Tensor([x0, y0]))
        refer_pts.append(torch.Tensor([x1, y1]))
    if len(query_pts) > 0:
        query_pts = torch.stack(query_pts).cuda()
        refer_pts = torch.stack(refer_pts).cuda()
    else:
        query_pts = torch.Tensor(0, 2).cuda()
        refer_pts = torch.Tensor(0, 2).cuda()

    return query_pts, refer_pts


def batch_get_mkpts(matches, query, refer, patch_size=8):
    query_pts = []
    refer_pts = []
    Hq, Wq = query.shape[2], query.shape[3]  # H, W
    Hr, Wr = refer.shape[2], refer.shape[3]  # H, W
    qcols = Wq // patch_size  # W_feat
    rcols = Wr // patch_size  # W_feat
    for pt in matches:
        # index = y * W_feat + x
        x0 = patch_size / 2 + (pt[1] % qcols) * patch_size
        y0 = patch_size / 2 + (pt[1] // qcols) * patch_size
        x1 = patch_size / 2 + (pt[2] % rcols) * patch_size
        y1 = patch_size / 2 + (pt[2] // rcols) * patch_size
        query_pts.append(torch.Tensor([pt[0], x0, y0]))
        refer_pts.append(torch.Tensor([pt[0], x1, y1]))
    if len(query_pts) > 0:
        query_pts = torch.stack(query_pts).cuda()
        refer_pts = torch.stack(refer_pts).cuda()
    else:
        query_pts = torch.Tensor(0, 3).cuda()
        refer_pts = torch.Tensor(0, 3).cuda()

    return query_pts, refer_pts


def rocket_score(mkpts0, mkpts1, query, refer, x, y, th=2):
    # ( ... 函数不变 ... )
    query = cv2.copyMakeBorder(query, 0, 800 - 512, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    out_img = np.concatenate([query, refer], axis=1).copy()
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    query_pts, refer_pts = query_pts[np.where((refer_pts - query_pts)[:, 1] > 0)[0]], refer_pts[
        np.where((refer_pts - query_pts)[:, 1] > 0)[0]]
    query_pts, refer_pts = query_pts[np.where((refer_pts - query_pts)[:, 1] < 289)[0]], refer_pts[
        np.where((refer_pts - query_pts)[:, 1] < 289)[0]]
    query_pts, refer_pts = query_pts[np.where((refer_pts - query_pts)[:, 0] > 0)[0]], refer_pts[
        np.where((refer_pts - query_pts)[:, 0] > 0)[0]]
    query_pts, refer_pts = query_pts[np.where((refer_pts - query_pts)[:, 0] < 289)[0]], refer_pts[
        np.where((refer_pts - query_pts)[:, 0] < 289)[0]]

    refer_pts[:, 0] = refer_pts[:, 0] + 512

    if len(refer_pts) == 0:
        return out_img, (0, 0)
    H, mask = cv2.findHomography(query_pts, refer_pts, cv2.RANSAC, ransacReprojThreshold=16)
    q_pts = []
    r_pts = []
    for i in range(query_pts.shape[0]):
        if mask[i]:
            cv2.line(out_img, (int(query_pts[i, 0]), int(query_pts[i, 1])),
                     (int(refer_pts[i, 0]), int(refer_pts[i, 1])), (0, 255, 0), 1)
            q_pts.append(query_pts[i])
            r_pts.append(refer_pts[i])
    query_pts, refer_pts = q_pts, r_pts
    query_pts = np.asarray(query_pts, np.float32)
    refer_pts = np.asarray(refer_pts, np.float32)
    try:
        r_x_means = int(np.mean((refer_pts)[:, 0], axis=0))
        r_y_means = int(np.mean((refer_pts)[:, 1], axis=0))
        q_x_means = int(np.mean((query_pts)[:, 0], axis=0))
        q_y_means = int(np.mean((query_pts)[:, 1], axis=0))
    except:
        r_x_means = r_y_means = q_x_means = q_y_means = 0

    real_x, real_y = r_x_means - q_x_means - 512, r_y_means - q_y_means
    cv2.rectangle(out_img, (real_x + 512, real_y), (real_x + 512 + 512, real_y + 512), (0, 255, 255), 2)
    print(x, y, "=>", real_x, real_y)
    return out_img, (real_x, real_y)


def rocket_match_num(mkpts0, mkpts1, query, refer, x, y, match_prec):
    # ( ... 函数不变 ... )
    query = cv2.copyMakeBorder(query, 0, 800 - 512, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    out_img = np.concatenate([query, refer], axis=1).copy()
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + 512
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    H, mask = cv2.findHomography(query_pts, refer_pts, cv2.RANSAC, ransacReprojThreshold=16)
    query_pts = [x for i, x in enumerate(query_pts) if mask[i]]
    refer_pts = [x for i, x in enumerate(refer_pts) if mask[i]]
    for pix_th in range(1, 11):
        q_pts = []
        r_pts = []
        for q_pt, r_pt in zip(query_pts, refer_pts):
            x0, y0 = q_pt
            x1, y1 = r_pt
            if np.abs(x1 - 512 - x0 - x) < pix_th and np.abs(y1 - y - y0) < pix_th:
                q_pts.append([x0, y0])
                r_pts.append([x1, y1])
                x_mean.append(x1 - 512 - x0 - x)
                y_mean.append(y1 - y - y0)

        match_prec[pix_th].append(len(q_pts) / len(query_pts))
    return match_prec


def rocket_match_num_with_gt(match_mask, mkpts0, mkpts1, query, refer, match_prec, patch_size=8):
    # ( ... 函数不变 ... )
    def get_gt_match(match_mask):
        grid = make_grid(match_mask.shape[1], match_mask.shape[0])
        _pts = grid[match_mask]
        query_pts = []
        refer_pts = []
        wq = query.shape[1]
        wr = refer.shape[1]
        qcols = wq // patch_size
        rcols = wr // patch_size
        for pt in _pts:
            x0 = patch_size / 2 + (pt[1] % qcols) * patch_size
            y0 = patch_size / 2 + (pt[1] // qcols) * patch_size
            x1 = patch_size / 2 + (pt[0] % rcols) * patch_size
            y1 = patch_size / 2 + (pt[0] // rcols) * patch_size
            query_pts.append([x0, y0])
            refer_pts.append([x1, y1])
        query_pts = np.asarray(query_pts, np.float32)
        refer_pts = np.asarray(refer_pts, np.float32)
        return query_pts, refer_pts

    out_img = np.concatenate([query, refer], axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    gt_query_pts, gt_refer_pts = get_gt_match(match_mask)
    if len(gt_query_pts) == 0:
        return match_prec
    for i in range(1, 11):
        mkpts11, mkpts10 = [], []
        if query_pts.shape[1] == 0:
            match_prec[i].append(0)
            continue
        for idx, mkp in enumerate(query_pts):
            m_gt_mkpts1 = gt_refer_pts[np.where((gt_query_pts == mkp).all(1))]
            if m_gt_mkpts1.shape[0] != 0:
                if np.linalg.norm(refer_pts[idx] - m_gt_mkpts1.squeeze()) < i:
                    mkpts10.append(query_pts[idx])
                    mkpts11.append(refer_pts[idx])

        if len(mkpts10) > 0:
            q_pts, r_pts = np.stack(mkpts10), np.stack(mkpts11)
        else:
            q_pts, r_pts = np.array([[]]), np.array([[]])
        match_prec[i].append(len(q_pts) / len(gt_query_pts))
    return match_prec


def draw_match_nir(mkpts0, mkpts1, query, refer, x, y, homo_filter=True, patch_size=8, th=1e3):
    # ( ... 函数不变 ... )
    out_img = np.concatenate([query, refer], axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    h_gt = None
    refer_pts[:, 0] = refer_pts[:, 0] + wq
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    for q_pt, r_pt in zip(query_pts, refer_pts):
        x0, y0 = q_pt
        x1, y1 = r_pt
        if np.abs(x1 - wq - x0 - x) < th and np.abs(y1 - y - y0) < th:
            q_pts.append([x0, y0])
            r_pts.append([x1, y1])
            x_mean.append(x1 - wq - x0 - x)
            y_mean.append(y1 - y - y0)

    if len(x_mean) != 0:
        x_means = int(np.mean(x_mean))
        y_means = int(np.mean(y_mean))
    else:
        x_means, y_means = 0, 0

    query_pts, refer_pts = q_pts, r_pts
    query_pts = np.asarray(query_pts, np.float32)
    refer_pts = np.asarray(refer_pts, np.float32)

    if homo_filter and query_pts.shape[0] > 4:
        H, mask = cv2.findHomography(query_pts, refer_pts, cv2.RANSAC, ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img, (int(query_pts[i, 0]), int(query_pts[i, 1])),
                         (int(refer_pts[i, 0]), int(refer_pts[i, 1])), (0, 255, 0), 1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img, (int(query_pts[i, 0]), int(query_pts[i, 1])),
                     (int(refer_pts[i, 0]), int(refer_pts[i, 1])), (0, 255, 0), 1)
    return out_img, h_gt


def eval_src_mma(mkpts0, mkpts1, query, refer, i_err, H_gt, thres=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], patch_size=8,
                 ransac_thres=3):
    # ( ... 函数不变 ... )
    mkpts0 = mkpts0.int().detach().cpu().numpy()
    mkpts1 = mkpts1.int().detach().cpu().numpy()
    H_pred = None
    if len(mkpts1) > 3:
        h_pred, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransacReprojThreshold=3)
        if len([mkpts0[x, :] for x in np.where(inliers > 0)[0]]) > 0:
            mkpts0 = np.stack([mkpts0[x, :] for x in np.where(inliers > 0)[0]])
            mkpts1 = np.stack([mkpts1[x, :] for x in np.where(inliers > 0)[0]])

    if len(mkpts1) == 0 or H_gt is None:
        dist = np.array([float("inf")])
    else:
        dist = cal_reproj_dists(mkpts0, mkpts1, H_gt)
    for thr in thres:
        i_err[thr] += np.mean(dist <= thr)
    return i_err, 1


def eval_src_homography(mkpts0, mkpts1, query, refer, H_gt, patch_size=8, ransac_thres=3):
    np.set_printoptions(suppress=True)
    mkpts0 = mkpts0.int().detach().cpu().numpy()
    mkpts1 = mkpts1.int().detach().cpu().numpy()
    h, w = query.shape[:2]

    # --- 修复 1: 当匹配点太少时, 返回 (nan, nan) ---
    if mkpts0.shape[0] < 4:
        return np.float64(np.nan), np.float64(np.nan)  # <-- 修复

    H_pred, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, ransacReprojThreshold=ransac_thres)

    # --- 健壮性检查: 确保 H_gt 存在 ---
    if H_gt is None:
        return np.float64(np.nan), np.float64(np.nan)

    # --- 修复 2: 当 H_pred 为 None 时, 返回 (nan, nan) ---
    if H_pred is None:
        return np.float64(np.nan), np.float64(np.nan)  # <-- 修复

    d_gt = np.abs(H_gt - H_pred).sum()

    d = mkpts1 - mkpts0
    dx = d[:, :1].mean()
    dy = d[:, 1:].mean()

    mk_point = np.ones(mkpts0.shape[0]).reshape(-1, 1)
    corners = np.concatenate((mkpts0, mk_point), axis=1)

    real_warped_corners = np.dot(corners, np.transpose(H_gt))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    warped_corners = np.dot(corners, np.transpose(H_pred))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))

    return corner_dist, d_gt


def draw_match_kpts(mkpts0, mkpts1, query, refer, x, y, homo_filter=True, patch_size=8, th=1e3):
    # ( ... 函数不变 ... )
    out_img = np.concatenate([query, refer], axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = mkpts0.int().detach().cpu().numpy()
    refer_pts = mkpts1.int().detach().cpu().numpy()
    refer_pts[:, 0] = refer_pts[:, 0] + wq
    q_pts = []
    r_pts = []
    x_mean = []
    y_mean = []
    for q_pt, r_pt in zip(query_pts, refer_pts):
        x0, y0 = q_pt
        x1, y1 = r_pt
        if np.abs(x1 - wq - x0 - x) < th and np.abs(y1 - y - y0) < th:
            q_pts.append([x0, y0])
            r_pts.append([x1, y1])

    query_pts, refer_pts = q_pts, r_pts
    query_pts = np.asarray(query_pts, np.float32)
    refer_pts = np.asarray(refer_pts, np.float32)

    if homo_filter and query_pts.shape[0] > 4:
        H, mask = cv2.findHomography(query_pts, refer_pts, cv2.RANSAC, ransacReprojThreshold=16)
        if len([x for x in range(len(mask)) if mask[x]]) > 5:
            pts = random.sample([(q_pts[i], r_pts[i]) for i in range(query_pts.shape[0]) if mask[i]], 5)
            for i in range(len(pts)):
                cv2.rectangle(out_img, (int(pts[i][0][0] - 8), int(pts[i][0][1] - 8)),
                              (int(pts[i][0][0] + 8), int(pts[i][0][1] + 8)), (0, 0, 255), 1)
                cv2.circle(out_img, (int(pts[i][0][0]), int(pts[i][0][1])), 2, (0, 255, 255), -1)
                cv2.rectangle(out_img, (int(pts[i][1][0] - 8), int(pts[i][1][1] - 8)),
                              (int(pts[i][1][0] + 8), int(pts[i][1][1] + 8)), (0, 0, 255), 1)
                cv2.circle(out_img, (int(pts[i][1][0]), int(pts[i][1][1])), 2, (0, 255, 255), -1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img, (int(query_pts[i, 0]), int(query_pts[i, 1])),
                     (int(refer_pts[i, 0]), int(refer_pts[i, 1])), (0, 255, 0), 1)
    return out_img