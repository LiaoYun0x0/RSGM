import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.Sigmoid()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class ResNetFPN_8_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']
        topic_num = config['topic_num']
        self.avgpool = nn.AdaptiveAvgPool1d(topic_num)
        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(3, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16

        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        x1 = torch.nn.functional.normalize(x1, dim=1, p=2)
        x2 = torch.nn.functional.normalize(x2, dim=1, p=2)
        x3 = torch.nn.functional.normalize(x3, dim=1, p=2)
        x4 = torch.nn.functional.normalize(x4, dim=1, p=2)
        x4_out = self.layer4_outconv(x4)
        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out + x4_out_2x)
        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)
        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)
        x1_out = torch.nn.functional.normalize(x1_out, dim=1, p=2)
        x2_out = torch.nn.functional.normalize(x2_out, dim=1, p=2)
        x3_out = torch.nn.functional.normalize(x3_out, dim=1, p=2)
        x4_out = torch.nn.functional.normalize(x4_out, dim=1, p=2)
        # x4 = torch.reshape(x3, (x3.shape[0], x3.shape[1], x3.shape[2] * x3.shape[3]))
        # x4 = self.avgpool(x4)  # [B, C, 1] # B,C,H*W
        # x4 = torch.transpose(x4, -1, -2)
        # x4 = self.norm_outlayer4(x4)
        # x4 = torch.transpose(x4, -1, -2)
        # print(x1_out.shape)
        # print(x2_out.shape)
        # print(x3_out.shape)
        # print(x4_out.shape)
        return [x4_out, x3_out,x2_out, x1_out]

class ResNetFPN_8_2_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim'] // 2
        block_dims = [dim // 2 for dim in config['block_dims']]
        topic_num = config['img_size']
        # self.avgpool = nn.AdaptiveAvgPool1d(topic_num)
        self.global_avg_pool1 = nn.AdaptiveAvgPool2d((topic_num//8, topic_num//8))
        self.global_avg_pool2 = nn.AdaptiveAvgPool2d((topic_num//16, topic_num//16))
        self.global_avg_pool3 = nn.AdaptiveAvgPool2d((topic_num//32, topic_num//32))
        self.global_avg_pool4 = nn.AdaptiveAvgPool2d((topic_num//64, topic_num//64))
        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )
        # self.downt_1 = conv1x1(256, 128)
        # self.downt_2 = conv1x1(392, 196)
        # self.downt_3 = conv1x1(512, 256)
        # self.downt_4 = conv1x1(512, 256)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        # print(x.shape)
        transform = transforms.Grayscale()
        x0 = transform(x[0]).unsqueeze(0)
        x1 = transform(x[1]).unsqueeze(0)
        x = torch.concat((x0, x1), 0)
        # print(image.shape)
        # weight = torch.tensor([0.299, 0.587, 0.114]).cuda()
        # x = torch.sum(x.permute(0, 2, 3, 1) * weight, dim=-1).unsqueeze(-1).permute(0, 3, 1, 2)
        # print(x.shape)
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        # print(x1.shape)
        x1 = torch.nn.functional.normalize(x1, dim=1, p=2)
        x2 = torch.nn.functional.normalize(x2, dim=1, p=2)
        x3 = torch.nn.functional.normalize(x3, dim=1, p=2)
        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)
        # print(x3.shape)
        x1_t = self.global_avg_pool4(x1_out)  # [B, C, 1] # B,C,H*W
        x2_t = self.global_avg_pool3(x2_out)  # [B, C, 1] # B,C,H*W
        x3_t = self.global_avg_pool2(x3_out)  # [B, C, 1] # B,C,H*W
        x4_t = self.global_avg_pool1(x3)  # [B, C, 1] # B,C,H*W
        # x4 = torch.flatten(self.global_avg_pool(x3), -2)  # [B, C, 1] # B,C,H*W
        x1_t = torch.reshape(x1_t, (x1_t.shape[0], x1_t.shape[1], x1_t.shape[2] * x1_t.shape[3])).permute(0, 2, 1)
        x2_t = torch.reshape(x2_t, (x2_t.shape[0], x2_t.shape[1], x2_t.shape[2] * x2_t.shape[3])).permute(0, 2, 1)
        x3_t = torch.reshape(x3_t, (x3_t.shape[0], x3_t.shape[1], x3_t.shape[2] * x3_t.shape[3])).permute(0, 2, 1)
        x4_t = torch.reshape(x4_t, (x4_t.shape[0], x4_t.shape[1], x4_t.shape[2] * x4_t.shape[3])).permute(0, 2, 1)

        x1_matrix = torch.einsum("nmd,nld->nml", x1_t[0].unsqueeze(0), x1_t[1].unsqueeze(0))
        x2_matrix = torch.einsum("nmd,nld->nml", x2_t[0].unsqueeze(0), x2_t[1].unsqueeze(0))
        x3_matrix = torch.einsum("nmd,nld->nml", x3_t[0].unsqueeze(0), x3_t[1].unsqueeze(0))
        x4_matrix = torch.einsum("nmd,nld->nml", x4_t[0].unsqueeze(0), x4_t[1].unsqueeze(0))
        # print(x1_matrix.shape)
        x1_matrix = F.softmax(x1_matrix, 1) * F.softmax(x1_matrix, 2)
        x1_matrix_idx = torch.argmax(x1_matrix, -1)
        x1_t = torch.concat((x1_t[0].unsqueeze(0), x1_t[1].unsqueeze(0)[:, x1_matrix_idx[-1]]), -1)

        x2_matrix = F.softmax(x2_matrix, 1) * F.softmax(x2_matrix, 2)
        x2_matrix_idx = torch.argmax(x2_matrix, -1)
        x2_t = torch.concat((x2_t[0].unsqueeze(0), x2_t[1].unsqueeze(0)[:, x2_matrix_idx[-1]]), -1)

        x3_matrix = F.softmax(x3_matrix, 1) * F.softmax(x3_matrix, 2)
        x3_matrix_idx = torch.argmax(x3_matrix, -1)
        x3_t = torch.concat((x3_t[0].unsqueeze(0), x3_t[1].unsqueeze(0)[:, x3_matrix_idx[-1]]), -1)

        # x4_matrix = F.softmax(x4_matrix, 1) * F.softmax(x4_matrix, 2)
        # x4_matrix_idx = torch.argmax(x4_matrix, -1)
        # x4_t = torch.concat((x4_t[0].unsqueeze(0), x4_t[1].unsqueeze(0)[:, x4_matrix_idx[-1]]), -1)
        # print(x4.shape)
        # x4 = torch.transpose(x4, -1, -2)
        # x4 = self.norm_outlayer4(x4)
        # x4 = torch.transpose(x4, -1, -2)
        # print(x3_out.shape)
        # return [x4, x3_out, x1_out]
        x1_t = torch.nn.functional.normalize(x1_t, dim=-1, p=2)
        x2_t = torch.nn.functional.normalize(x2_t, dim=-1, p=2)
        x3_t = torch.nn.functional.normalize(x3_t, dim=-1, p=2)
        # print(x1_t.shape)
        # print(x2_t.shape)
        # print(x3_t.shape)
        # print(x4_t.shape)
        # torch.Size([1, 25, 64])
        # torch.Size([1, 100, 96])
        # torch.Size([1, 400, 128])
        # torch.Size([1, 1600, 256])
        return [x1_t, x2_t, x3_t]

class ResNetFPN_16_4(nn.Module):
    """
    ResNet+FPN, output resolution are 1/16 and 1/4.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16

        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )

        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        # FPN
        x4_out = self.layer4_outconv(x4)

        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out + x4_out_2x)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

        return [x4_out, x2_out]
