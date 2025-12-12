import os
import os.path as osp
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import re
import csv

import opt


def extra_features_map():
    map = {}
    csv_file = osp.join(opt.ROOT, "extra.csv")
    npz_file_dir = osp.join(opt.ROOT, "extra_features")
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            file_name = row['file_name']
            feature_path = osp.join(npz_file_dir, row['npz_path'])

            # 载入 npz
            try:
                npz = np.load(feature_path, allow_pickle=False)
            except Exception as e:
                print(f"加载 npz 失败：{e}")
                continue

            # keys 排序（假设为 ch1..ch6）
            keys = sorted(list(npz.files))
            # 检查至少 1 个 key
            if len(keys) == 0:
                print(f"{feature_path} 没有可用的 keys")
                continue

            channels = [npz[k].astype(np.float32) for k in keys]
            # 确保每个 channel 形状一致
            shapes = [c.shape for c in channels]
            if not all(s == shapes[0] for s in shapes):
                print(f"警告：{feature_path} 中 ch 的形状不一致: {shapes}")
            resized_channels = []
            for c in channels:  # c 的形状是 (H, W)
                if(c.shape[0] != c.shape[1]):
                    c = pad_to_square(c)
                basic_size = opt.input_size
                resized = cv2.resize(c, (basic_size[1], basic_size[0]), interpolation=cv2.INTER_LINEAR)
                resized_channels.append(resized)
            stacked = np.stack(resized_channels, axis=0)  # (6, H, W) 或 (6, dim) 视原数据而定
            map[file_name] = stacked

    return map


def readfile(path, label, basic_size=None, extra=False):
    if basic_size is None:
        basic_size = opt.input_size

    # 统计图片数量（只统计文件，不递归子文件夹）
    count = 0
    for _, _, files in os.walk(path):
        count += len(files)

    # 预分配图像和标签（extra 先用 list 收集）
    x = np.zeros((count, basic_size[0], basic_size[1], 3), dtype=np.uint8)
    y = np.zeros(count, dtype=np.uint8) if label else None
    extra_list = [] if extra else None

    i = 0
    type_names = sorted(os.listdir(path))
    extra_feature_dict = extra_features_map() if extra else {}

    for type_name in type_names:
        img_dir = os.path.join(path, type_name)
        if not osp.isdir(img_dir):
            continue
        files = os.listdir(img_dir)
        files = sorted(files, key=natural_sort_key)
        for j, file in enumerate(files):
            img_path = os.path.join(img_dir, file)
            img = cv2.imread(img_path)
            if img.shape[0] != img.shape[1]:
                img = pad_to_square(img)
            if img is None:
                print(f"无法读取图片: {img_path}, 跳过")
                continue

            img_resized = cv2.resize(img, (basic_size[1], basic_size[0]))  # cv2.resize uses (w,h)
            x[i, :, :, :] = img_resized

            if extra:
                # 计算 f_key
                f_key = "-".join(osp.splitext(file)[0].split('-')[:-1])

                if f_key in extra_feature_dict:
                    f_value = extra_feature_dict[f_key]  # numpy array shape (6, ...)
                    extra_list.append(f_value)
                else:
                    print(f'{f_key} has not extra feature！填充 zeros')
                    # 以第一个已有样本的形状填充 zeros，如果没有已知形状则填充空数组
                    if len(extra_list) > 0:
                        shape = extra_list[0].shape
                        extra_list.append(np.zeros(shape, dtype=np.float32))
                    else:
                        # 没有任何样本可参考，先 append None，后面会剔除或处理
                        extra_list.append(None)

            if label:
                # 如果 type_name 不是在 opt.cls 中会抛出异常
                y[i] = int(opt.cls.index(type_name))
            i += 1
    if label and extra:
        return x, y, np.stack(extra_list, axis=0)
    elif label:
        return x, y, None
    else:
        return x


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None, extra=None):
        self.x = x
        self.y = y
        self.extra = extra
        if y is not None:
            # label required to be LongTensor
            self.y = torch.LongTensor(y)
        if extra is not None:
            # convert extra to float tensor (N, 6, ...)
            self.extra = torch.from_numpy(np.array(extra)).float()
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.extra is not None and self.y is not None:
            Y = self.y[index]
            return X, Y, self.extra[index]
        elif self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


def natural_sort_key(s):
    """
    按文件名的结构排序，即依次比较文件名的非数字和数字部分
    """
    # 将字符串按照数字和非数字部分分割，返回分割后的子串列表
    sub_strings = re.split(r'(\d+)', s)
    # 如果当前子串由数字组成，则将它转换为整数；否则返回原始子串
    sub_strings = [int(c) if c.isdigit() else c for c in sub_strings]
    # 根据分割后的子串列表以及上述函数的返回值，创建一个新的列表
    # 按照数字部分从小到大排序，然后按照非数字部分的字典序排序
    return sub_strings


def pad_to_square(arr):
    """
    极简方式：创建w×w全0数组，将原数组直接赋值到对应位置（原索引不变）
    """
    if len(arr.shape) > 2:
        h, w, c = arr.shape
    else:
        h, w = arr.shape
    if not w > h:
        raise ValueError("宽度必须大于高度（w > h）")

    if len(arr.shape) > 2:
        # 1. 创建w×w的全0正方形数组
        square_arr = np.zeros((w, w, c), dtype=arr.dtype)
        # 2. 将原数组赋值到正方形数组的前h行（行索引0~h-1），列索引不变
        square_arr[:h, :w, :] = arr  # 核心：原索引完全不变
    else:
        square_arr = np.zeros((w, w), dtype=arr.dtype)
        square_arr[:h, :w] = arr  # 核心：原索引完全不变
    return square_arr


if __name__ == '__main__':
    datasets_path = r'E:\datasets\PAUT\datasets-s'

    test_x, test_y, test_extra = readfile(os.path.join(datasets_path, "testing"), True, extra=True)
    print("Size of testing data = {}".format(len(test_x)))

    print("test_x shape:", test_x.shape)
    print("test_y shape:", test_y.shape)
    if test_extra is not None:
        print("extra shape:", test_extra.shape)  # 期望: (N, 6, ...)
