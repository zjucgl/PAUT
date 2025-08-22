# Import需要的套件
import os
import os.path as osp
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import re
import csv

import opt


def pre(raw):
    gray_image = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    # 使用中值滤波器对图像进行滤波， 对于图像中孤立的噪声点有很大的促进作用，并且会使图像变得清晰
    filtered_image = cv2.medianBlur(gray_image, ksize=3)
    # 使用均值滤波进行降噪，使平滑
    filtered_image = cv2.blur(filtered_image, (3, 3))
    # 直方图均衡化增加对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    filtered_image = clahe.apply(filtered_image)
    res = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
    return res


def get_former_slice(dir_path, file):
    file_names = os.path.splitext(file)
    names = file_names[0].split('_')
    slice_index = int(names[-1])
    if slice_index == 0:
        return None
    for i in range(1, slice_index):
        former_index = slice_index - i
        former_file = f"{'_'.join(names[:-1])}_{former_index}{file_names[1]}"
        former_slice = os.path.join(dir_path, former_file)
        if os.path.exists(former_slice):
            return former_slice
    return None


# Read image 利用 OpenCV(cv2) 读入照片并存放在 numpy array 中
def readfile(path, label, basic_size=None, former=False, ifSide=False, extra=False):
    if basic_size is None:
        basic_size = opt.input_size
    # label 是一个 boolean variable, 代表需不需要回传 y 值
    # 获得路径下所有图片的的数量
    count = 0
    for _, _, files in os.walk(path):
        count += len(files)

    x = np.zeros((count, basic_size[0], basic_size[1], 3), dtype=np.uint8)
    y = np.zeros(count, dtype=np.uint8)
    # former_slice = np.zeros((count, basic_size[0], basic_size[1], 3), dtype=np.uint8) if former else None
    former_slice = np.zeros((count, basic_size[0], basic_size[1]), dtype=np.uint8) if former else None
    sides = []
    file_names = []
    extra_features = np.zeros(count, dtype=np.uint8)

    # 图片下标
    i = 0
    type_names = os.listdir(path)

    extra_feature_dict = {}
    if extra:
        csv_file = osp.join(opt.ROOT, "feature.csv")
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_path = row['image_path']
                feature_value = row['feature_value']
                extra_feature_dict[image_path] = feature_value
    # print(extra_feature_dict)
    for type_name in type_names:
        img_dir = os.path.join(path, type_name)
        files = os.listdir(img_dir)
        files = sorted(files, key=natural_sort_key)
        for j, file in enumerate(files):
            img = cv2.imread(os.path.join(img_dir, file))  # os.path.join(path, file) 路径名合并
            # img = pre(img)
            # print(img_dir, file)
            # print(os.path.join(img_dir, file), img.shape)
            x[i, :, :] = cv2.resize(img, basic_size)
            # if i > 0 and if_side_slice(files[i - 1], files[i]):
            #     side.append(-1)
            if extra:
                if file in extra_feature_dict:
                    extra_features[i] = extra_feature_dict[file]
                else:
                    print(f'{file} has not extra feature！')
                    extra_features[i] = 0
            if ifSide:
                if j > 1 and if_side_slice(files[j - 1], files[j]):
                    sides.append(1)
                else:
                    sides.append(0)
                file_names.append(file)
            if former:
                root = '/home/admin/datasets/PAUT/datasets-extra'
                f_path = os.path.splitext(file)
                f_name = '-'.join(f_path[0].split('-')[:-1])
                f_name = f"{f_name}{f_path[1]}"
                former_path = os.path.join(root, f_name)
                # former_path = get_former_slice(img_dir, file)
                # Wavelet = opt.workspace_dir.split("/")[-1].split("-")[-1]
                # former_path = os.path.join(img_dir.replace('datasets-s', f'datasets-a-wav-{Wavelet}'), file)
                # 如果存在
                if former_path:
                    # former_img = cv2.imread(os.path.join(img_dir, former_path))
                    # former_slice[i, :, :] = cv2.resize(former_img, basic_size)
                    former_img = cv2.imread(os.path.join(img_dir, former_path), 0)
                    former_slice[i] = cv2.resize(former_img, (224, 224))
                else:
                    print(f"{former_path} 文件不存在!")
            if label:
                y[i] = int(opt.cls.index(type_name))
            i += 1
    if label and ifSide:
        return x, y, former_slice, sides, file_names
    elif label and extra:
        return x, y, extra_features
    elif label:
        return x, y, None
    else:
        return x


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None, former_slice=None, sides=None, extra=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        self.sides = sides
        self.former_slice = former_slice
        self.extra = extra
        if y is not None:
            self.y = torch.LongTensor(y)
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
        elif self.sides is not None:
            Y = self.y[index]
            return X, Y, self.sides[index]
        elif self.y is not None:
            Y = self.y[index]
            return X, Y
        else:  # 如果没有标签那么只返回X
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


# 通过两个切片的文件名，判断切片是否相邻
def if_side_slice(slice_x: str, slice_y: str):
    slice_x = os.path.splitext(slice_x)[0]
    slice_y = os.path.splitext(slice_y)[0]
    x_name = '_'.join(slice_x.split('_')[:-1])
    x_index = slice_x.split('_')[-1]
    y_name = '_'.join(slice_y.split('_')[:-1])
    y_index = slice_y.split('_')[-1]
    return x_name == y_name and int(x_index) + 1 == int(y_index)


if __name__ == '__main__':
    # dir_path = r'E:\datasets\gydp\classification\raw\PO'
    # file = 'flip_PO_1_0.jpg'
    # print(get_former_slice(dir_path, file))

    # path = 'E:/datasets/gydp/classification/datasets1/testing'
    # test_x, test_y, test_former, sides = readfile(path, True, former=False)
    # print("Size of test data = {}".format(len(test_x)))
    # # testing 时不需做 data augmentation
    # test_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # test_set = ImgDataset(test_x, test_y, transform=test_transform, former_slice=test_former, sides=sides)
    # test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    # for data in test_loader:
    #     # print(len(data))
    #     print(data[1], data[-1])
    filename = '29-1200-S4-00001-P1-100.png'
    path = os.path.splitext(filename)
    filename = '-'.join(path[0].split('-')[:-1])
    filename = f"{filename}{path[1]}"
    print(filename)