import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import cv2


def get_inputs_list(dataset_dir):
    input_list = []
    file_name_list = []

    for root, dirs, names in os.walk(dataset_dir):
        for name in names:
            input_list.append(os.path.join(root, name))
            file_name_list.append(name)
    input_list.sort()
    file_name_list.sort()
    return input_list, file_name_list


class TrainDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.dataset_dir = args.dataset_dir + mode + '/'
        self.crop_patch = False
        self.patch_size = args.patch_size

        self.gt_dir = os.path.join(self.dataset_dir, 'gt/')
        self.low_dir = os.path.join(self.dataset_dir, 'input/')

        self.gt_list, self.gt_names = get_inputs_list(self.gt_dir)
        self.input_list, self.low_names = get_inputs_list(self.low_dir)

    def __getitem__(self, item):
        input_img = Image.open(self.input_list[item]).convert('RGB')
        # input_img = input_img.resize((self.patch_size, self.patch_size), Image.ANTIALIAS)
        lr = np.asarray(input_img).astype(np.float32) / 255.0  # (h, w, c)

        gt_img = Image.open(self.gt_list[item]).convert('RGB')
        # gt_img = gt_img.resize((self.patch_size, self.patch_size), Image.ANTIALIAS)
        gt = np.asarray(gt_img).astype(np.float32) / 255.0  # (h, w, c)

        h, w, _ = lr.shape
        """ crop pair patch """
        x = np.random.randint(0, h - self.patch_size + 1)
        y = np.random.randint(0, w - self.patch_size + 1)
        gt_patch = gt[x: x + self.patch_size, y: y + self.patch_size, :]
        lr_patch = lr[x: x + self.patch_size, y: y + self.patch_size, :]

        """ resize """
        # lr_patch, gt_patch = lr, gt

        # augment
        train_data, train_gt = self.augment(lr_patch, gt_patch)

        # r, g, b = train_data[:, :, 0] + 1, train_data[:, :, 1] + 1, train_data[:, :, 2] + 1
        # luminance_map = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.

        train_data = T.ToTensor()(train_data)  # (c, h, w)
        train_gt = T.ToTensor()(train_gt)
        # attn_map = T.ToTensor()(luminance_map)

        return train_data, train_gt

    def __len__(self):
        return len(self.input_list)

    def augment(self, lr, hr, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot180 = rot and random.random() < 0.5

        def _augment(lr_img, hr_img):
            if hflip:
                lr_img = lr_img[:, ::-1, :]
                hr_img = hr_img[:, ::-1, :]
            if vflip:
                lr_img = lr_img[::-1, :, :]
                hr_img = hr_img[::-1, :, :]
            if rot180:
                lr_img = np.rot90(lr_img, 2)
                hr_img = np.rot90(hr_img, 2)

            lr_img = np.ascontiguousarray(lr_img.copy())
            hr_img = np.ascontiguousarray(hr_img.copy())

            return lr_img, hr_img

        lr, hr = _augment(lr, hr)
        return lr, hr


class TestDataset(Dataset):
    def __init__(self, args, mode='val', crop_border=False, crop_patch=False):
        self.dataset_dir = args.dataset_dir + mode + '/'
        self.gt_dir = os.path.join(self.dataset_dir, 'gt/')  # BAID-resize: gt, BAID-part: output, MSEC: labels
        self.low_dir = os.path.join(self.dataset_dir, 'input/')  # MSEC: inputs

        self.gt_list, self.gt_names = get_inputs_list(self.gt_dir)
        self.input_list, self.low_names = get_inputs_list(self.low_dir)

        self.crop_border = crop_border
        self.crop_patch = crop_patch
        self.patch_size = 512

    def __getitem__(self, item):
        gt = Image.open(self.gt_list[item]).convert('RGB')
        lr_input = Image.open(self.input_list[item]).convert('RGB')

        lr_t = np.asarray(lr_input).astype(np.float32) / 255.0  # (h, w, c)
        gt_t = np.asarray(gt).astype(np.float32) / 255.0  # (h, w, c)


        lr_t = T.ToTensor()(lr_t)
        gt_t = T.ToTensor()(gt_t)


        if self.crop_border:

            h, w = lr_t.shape[1], lr_t.shape[2]
            new_h, new_w = int(h - h % 8), int(w - w % 8)
            gt_t = gt_t[:, :new_h, :new_w]
            lr_t = lr_t[:, :new_h, :new_w]

        if self.crop_patch:
            lr_t = T.Resize(size=(self.patch_size, self.patch_size))(lr_t)
            gt_t = T.Resize(size=(self.patch_size, self.patch_size))(gt_t)
            # lr_t = T.CenterCrop(size=(self.patch_size, self.patch_size))(lr_t)
            # gt_t = T.CenterCrop(size=(self.patch_size, self.patch_size))(gt_t)
            # x = np.random.randint(0, h - self.patch_size + 1)
            # y = np.random.randint(0, w - self.patch_size + 1)
            # gt_t = gt_t[:, x: x + self.patch_size, y: y + self.patch_size]
            # lr_t = lr_t[:, x: x + self.patch_size, y: y + self.patch_size]
        return lr_t, gt_t, self.low_names[item]

    def __len__(self):
        return len(self.input_list)