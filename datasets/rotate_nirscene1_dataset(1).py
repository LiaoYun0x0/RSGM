import cv2
import os

from torch import dtype
from torch.utils.data import Dataset, DataLoader
import random
import torch
import numpy as np
from .utils import *
from skimage import io, color


class RotateNIRDataset(Dataset):
    def __init__(self, data_file, size=(320, 320), stride=8, aug=True):
        self.data_file = data_file
        with open(data_file, 'r') as f:
            self.train_data = f.readlines()

        self.size = size
        self.aug = aug
        self.stride = stride  # for generating gt-mask needed to compute local-feature loss
        self.query_pts = self._make_query_pts()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def _read_file_paths(self, data_dir):
        assert os.path.isdir(data_dir), "%s should be a dir which contains images only" % data_dir
        file_paths = os.listdir(data_dir)
        return file_paths

    def __getitem__(self, index: int):
        opt, sar = self.train_data[index].strip('\n').split(' ')
        opt_img_path = os.path.join(os.path.dirname(self.data_file), '', opt)
        opt_img = io.imread(opt_img_path.replace('stage1_', ''))
        # opt_img = color.rgba2rgb(opt_img)
        h, w, c = opt_img.shape

        sar_img_path = os.path.join(os.path.dirname(self.data_file), '', sar)
        sar_img = io.imread(sar_img_path.replace('stage1_', ''))
        sar_img = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2RGB)
        # 裁剪的图，转换矩阵，和图片起始坐标
        query, refer, Mr, Mq, qc, rc, H_gt = self._generate_ref(opt_img, sar_img)
        # print(query.shape)
        # print(Mr)
        # print(Mq)
        # print(Mr-Mq)

        # dropout query
        label_matrix = self._generate_label(Mr, Mq, qc, rc, (int(0), int(0)))  # 400x400
        # print(H_gt)
        # x, y = np.nonzero(label_matrix)
        # x_x = x // 40
        # x_y = x % 40
        # y_x = y // 40
        # y_y = y % 40
        # x_index = np.array(list(zip(x_x, x_y)))
        # y_index = np.array(list(zip(y_x, y_y)))
        # h_gt = None
        # if x_index.shape[0] <=6:
        #     h_gt = np.eye(3)
        # else:
        #     h_gt, mask = cv2.findHomography(x_index, y_index, cv2.RANSAC, ransacReprojThreshold=3)
        # h_gt = h_gt if h_gt is not None else np.eye(3)
        # h_gt = torch.Tensor(h_gt)
        # print(h_gt)
        # print(x.shape, y.shape)
        # cv2.imshow("query:", query)
        # cv2.imshow("refer:", refer)
        # cv2.waitKey()
        query = query.transpose(2, 0, 1)
        refer = refer.transpose(2, 0, 1)

        query = ((query / 255.0) - self.mean) / self.std
        refer = ((refer / 255.0) - self.mean) / self.std

        sample = {
            "refer": refer,
            "query": query,
            "gt_matrix": label_matrix,
            "h_gt": H_gt,
            "M": Mr
            # "M": M,
            # "Mr": Mr,
            # "Mq": Mq
        }
        return sample

    def _generate_ref(self, refer, query):
        """
        通过sar和optical找到相对应的映射关系矩阵
        """
        H_gt = None
        # 1/3的概率。同时数据增强为真
        if random.sample([0, 1, 2], 1)[0] != 0 and self.aug is True:
            # query裁剪成320*320 3*3对角矩阵，图片起始坐标
            crop_query, crop_M_query, qc = self._random_crop(query)
            # print(crop_query.shape) 320 320
            # print(query.shape) 原始
            # 数据增强，旋转。翻转。加噪
            query, Mq_ = self._aug_img(crop_query, query, qc)  # 320x320x3, 3x3
            Mq = np.matmul(Mq_, crop_M_query)  # 转换矩阵= 裁剪+增强
            # print(Mq_)
            crop_refer, crop_M_refer, rc = self._random_crop(refer)
            refer, Mr_ = self._aug_img(crop_refer, refer, rc)
            Mr = np.matmul(Mr_, crop_M_refer)
            # print(Mr_)
            cos_theta1, sin_theta1 = Mq_[0, 0], Mq_[0, 1]
            cos_theta2, sin_theta2 = Mr_[0, 0], Mr_[0, 1]
            theta1 = np.arctan2(sin_theta1, cos_theta1)
            theta2 = np.arctan2(sin_theta2, cos_theta2)
            theta_rel = (theta1 - theta2) % (2 * np.pi) - np.pi  # 调整到[-pi, pi]范围
            if theta_rel < -np.pi:
                theta_rel += 2 * np.pi
            theta_rel = np.degrees(theta_rel)
            R_rel = np.array([
                [np.cos(theta_rel), -np.sin(theta_rel)],
                [np.sin(theta_rel), np.cos(theta_rel)]
            ])
            T_rel = np.hstack((R_rel, np.zeros((2, 1))))  # 先水平堆叠旋转部分和零列
            T_rel = np.vstack((T_rel, [0, 0, 1]))  # 然后垂直堆叠最后一行
            # H_gt_ = T_rel
            H_gt = (crop_M_query - crop_M_refer)+np.eye(3)
            H_gt = np.matmul(T_rel, H_gt)  # 转换矩阵= 裁剪+增强
            # print(H_gt_)
            # print(H_gt)
        else:
            crop_query, crop_M_query, qc = self._random_crop2_1(query)
            query, Mq = self._aug_img(crop_query, query, qc, -1)  # 320x320x3, 3x3
            Mq = np.matmul(Mq, crop_M_query)
            # print(Mq)
            crop_refer, crop_M_refer, rc = self._random_crop2_2(refer)
            # print(crop_refer.shape)
            crop_refer = cv2.copyMakeBorder(crop_refer, 0, 32, 0, 32, cv2.BORDER_CONSTANT, value=[128, 128, 128])
            # print(crop_refer.shape)
            refer, Mr = self._aug_img(crop_refer, refer, rc, -1)
            Mr = np.matmul(Mr, crop_M_refer)
            H_gt = np.eye(3) + (Mq - Mr)
            # print(Mr)

        # print(query.shape)
        # print(refer.shape)
        # print(Mr.shape)
        # print(Mq.shape)
        # print(qc.shape)
        # print(rc.shape)
        # H_gt = np.eye(3)+(Mr-Mq)

        # 裁剪的图，以及裁剪图的转换矩阵，和裁剪的起始坐标
        return query, refer, Mr, Mq, qc, rc, H_gt
    def _random_crop2_1(self, img):
        h, w, c = img.shape

        # matrix = np.eye(3)
        # x, y = random.randint(0, w - 320), random.randint(0, h - 320)
        x,y = 0, 0
        img = img[y:512 + y, x:512 + x]

        crop_M = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])
        # img = cv2.resize(img, (320, 320))
        return img, crop_M, (x, y)
    def _random_crop2_2(self, img):
        h, w, c = img.shape

        # matrix = np.eye(3)
        # x, y = random.randint(0, w - 320), random.randint(0, h - 320)
        x,y = 32, 32
        img = img[y:512, x:512]

        crop_M = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])
        # img = cv2.resize(img, (320, 320))
        return img, crop_M, (x, y)
    #
    # 转换矩阵，起始坐标 coor=(0,0)
    def _generate_label(self, Mr, Mq, qc, rc, coor, drop_mask=True):
        """
        M random_place
        Mr aug_refer
        Mq aug_query
        """
        # 1/8,1/8
        ncols, nrows = self.size[0] // self.stride, self.size[1] // self.stride

        label = np.zeros((ncols * nrows, ncols * nrows), dtype=np.int16)  # (1600, 1600)

        Mq_inv = np.linalg.inv(Mq)  # 转换矩阵的逆矩阵
        # print(Mq_inv.shape) 3*3
        # print( self.query_pts.T.shape)# 3 * 1600
        src_pts = np.matmul(Mq_inv, self.query_pts.T)  # self.query_pts (3x1600) , shape:40x40x3, 变换位置
        # print(src_pts.shape) # 3*1600
        # mask0 = (0 <= src_pts[0, :]) & (src_pts[0, :] < 320) & (0 <= src_pts[1, :]) & (src_pts[1, :] < 320)

        # sar原图平移
        trans_M = np.array([
            [1, 0, coor[0]],
            [0, 1, coor[1]],
            [0, 0, 1]
        ])
        refer_pts = np.matmul(trans_M, src_pts)
        # 平移得到sar和opt对其
        trans_M1 = np.array([
            [1, 0, qc[0]],
            [0, 1, qc[1]],
            [0, 0, 1]
        ])
        trans_M2 = np.array([
            [1, 0, qc[0] - rc[0]],
            [0, 1, qc[1] - rc[1]],
            [0, 0, 1]
        ])
        # H_gt = np.array([
        #     [1, 0, rc[0] - qc[0]],
        #     [0, 1, qc[1] - rc[1]],
        #     [0, 0, 1]
        # ])
        trans_M = np.matmul(trans_M2, trans_M1)
        trans_M3 = np.array([
            [1, 0, -rc[0]],
            [0, 1, -rc[1]],
            [0, 0, 1]
        ])
        trans_M = np.matmul(trans_M3, trans_M)
        # print(trans_M)
        refer_pts = np.matmul(trans_M, refer_pts)
        # print(src_pts.shape)
        # print(refer_pts.shape)
        # opt原图裁剪
        # index_x = qc[0] - rc[0]
        # index_y = qc[1] - rc[1]
        # print(index_x)
        # print(index_y)
        # xx = np.arange(0, 1600)
        # yy = np.arange(0, 1600)
        # xx_x = xx // 40
        # xx_y = xx % 40
        # yy_x = yy // 40
        # yy_y = yy % 40
        # yy_x += index_x
        # yy_y += index_y
        # x_index = np.array(list(zip(xx_x, xx_y)))
        # y_index = np.array(list(zip(yy_x, yy_y)))
        # print(x_index, y_index)
        # yy = yy.reshape(nrows, ncols)
        # H, mask = cv2.findHomography(x_index, y_index, cv2.RANSAC, ransacReprojThreshold=3)
        # print(H)
        refer_pts = np.matmul(Mr, refer_pts)
        # print(refer_pts.shape) # 3 * 1600
        mask1 = (0 <= refer_pts[0, :]) & (refer_pts[0, :] < 320) & (0 <= refer_pts[1, :]) & (refer_pts[1, :] < 320)

        mask = mask1  # (1600,)
        # 坐标成1/8
        match_index = np.int16(refer_pts[0, :] // self.stride + (refer_pts[1, :] // self.stride) * ncols)  # (1600,)
        # print(match_index.shape)
        indexes = np.arange(nrows * ncols)[mask]  # 1600 个坐标中有效的几个
        # print(indexes.shape)
        for index in indexes:
            label[index][match_index[index]] = 1
        return label

    #  对应关系的坐标值矩阵和默认值附一
    def _make_query_pts(self):
        ncols, nrows = self.size[0] // self.stride, self.size[1] // self.stride
        half_stride = (self.stride - 1) / 2
        # print(half_stride) 3.5
        xs = np.arange(ncols)
        ys = np.arange(nrows)
        xs = np.tile(xs[np.newaxis, :], (nrows, 1))
        # print(xs.shape)
        ys = np.tile(ys[:, np.newaxis], (1, ncols))
        ones = np.ones((nrows, ncols, 1), dtype=np.int16)  # 40 * 40 全1
        grid = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis], ones], axis=-1)
        # print(grid.shape)
        grid[:, :, :2] = grid[:, :, :2] * self.stride + half_stride  # (0:20, 0:20, 1) , shape:20x20x3
        return grid.reshape(-1, 3)  # (nrows*ncols , 3)

    # 随机标志
    def _random_flag(self, thresh=-1):
        return np.random.rand(1) < thresh

    #
    def _random_crop(self, img):
        h, w, c = img.shape
        # 320 320 3
        # print(img.shape)
        # matrix = np.eye(3) 对角矩阵
        # 随机生成70到70的随机数
        # x, y = random.randint(70, max(w - 460, 70)), random.randint(70, max(h - 460, 70))
        x, y = random.randint(0, w - 320), random.randint(0, h - 320)
        # x,y = 70, 70
        # x,y = 0, 3
        img = img[y:320 + y, x:320 + x]
        # 裁剪出320*320的图
        crop_M = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])
        # img = cv2.resize(img, (320, 320))
        # img 裁剪的图片 crop_M对角矩阵，x,y图片起始坐标
        return img, crop_M, (x, y)

    #
    def _random_crop2(self, img):
        h, w, c = img.shape

        # matrix = np.eye(3)
        x, y = random.randint(0, w - 320), random.randint(0, h - 320)
        # x,y = 0, 3
        img = img[y:320 + y, x:320 + x]

        crop_M = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])
        # img = cv2.resize(img, (320, 320))
        return img, crop_M, (x, y)

    #
    def _aug_img(self, img, src, qc, aug=1):
        # img裁剪大小 src原图的大小 qc裁剪起始坐标
        h, w = img.shape[:2]
        matrix = np.eye(3)  # 3*3的对角矩阵

        if self._random_flag(aug):  # 1<传值（-1）真
            img, rM = random_rotation2(img, src, qc, max_degree=60)
            # img,rM = random_rotation(img, max_degree=45)
            rM = np.concatenate([rM, np.array([[0, 0, 1]], np.float32)])
            matrix = np.matmul(rM, matrix)  # 旋转矩阵

        if self._random_flag():  # 假
            kernel = random.choice([1, 3, 5, 7])
            img = blur_image(img, kernel)

        if self._random_flag(aug * 0.2):  # 假
            img = img[:, ::-1, ...].copy()  # horizontal flip 水平翻转
            fM = np.array([
                [-1, 0, w - 1],
                [0, 1, 0],
                [0, 0, 1]
            ], np.float32)
            matrix = np.matmul(fM, matrix)

        if self._random_flag(aug * 0.2):  # 假
            img = img[::-1, :, ...].copy()  # vertical flip 垂直翻转
            vfM = np.array([
                [1, 0, 0],
                [0, -1, h - 1],
                [0, 0, 1]
            ], np.float32)
            matrix = np.matmul(vfM, matrix)

        if self._random_flag():
            img = random_gauss_noise(img)

        return img, matrix

    #
    def __len__(self):
        return len(self.train_data)


def build_Rotate_NIR(
        train_data_file,
        test_data_file,
        size,
        stride):
    train_data = RotateNIRDataset(
        train_data_file,
        size=(320, 320),
        stride=8,
        aug=False)
    test_data = RotateNIRDataset(
        test_data_file,
        size=(512, 512),
        stride=8,
        aug=False)

    return train_data, test_data


if __name__ == "__main__":
    from utils import _transform_inv, draw_match

    size = (320, 320)
    dataloader = DataLoader(
        RotateNIRDataset("/home/ly/Documents/zkj/dataset/nirscene1/train.txt", size=size, aug=True),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True)
    print(len(dataloader))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    check_index = 0
    num = 0
    while 1:
        for sample in dataloader:
            query, refer, label_matrix = sample["query"], sample["refer"], sample["gt_matrix"]
            query0 = query.detach().cpu().numpy()[check_index]
            refer0 = refer.detach().cpu().numpy()[check_index]
            label_matrix0 = label_matrix.detach().cpu().numpy()[check_index]
            query1 = query.detach().cpu().numpy()[check_index + 1]
            refer1 = refer.detach().cpu().numpy()[check_index + 1]
            label_matrix1 = label_matrix.detach().cpu().numpy()[check_index + 1]

            sq0 = _transform_inv(query0, mean, std)
            sr0 = _transform_inv(refer0, mean, std)
            out0 = draw_match(label_matrix0 > 0, sq0, sr0).squeeze()
            sq1 = _transform_inv(query1, mean, std)
            sr1 = _transform_inv(refer1, mean, std)
            out1 = draw_match(label_matrix1 > 0, sq1, sr1).squeeze()
            cv2.imwrite(f"images/match_img0_{num}.jpg", out0)
            cv2.imwrite(f"images/match_img1_{num}.jpg", out1)
            num = num + 1
