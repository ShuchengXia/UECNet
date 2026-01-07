import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import cv2

def get_image_list(folder):
    image_list = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp')):
                image_list.append(os.path.join(root, name))
    return image_list

def find_gt_path(gt_dir, base_name):
    # 自动尝试多种扩展名
    extensions = ['.JPG', '.jpg', '.png', '.PNG']
    for ext in extensions:
        gt_path = os.path.join(gt_dir, base_name + ext)
        if os.path.exists(gt_path):
            return gt_path
    return None


class TrainDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.dataset_dir = os.path.join(args.dataset_dir, mode)
        self.crop_patch = True
        self.patch_size = args.patch_size

        self.input_dir = os.path.join(self.dataset_dir, 'input')
        self.gt_dir = os.path.join(self.dataset_dir, 'gt')

        self.input_list = get_image_list(self.input_dir)
        self.paired_list = []

        for input_path in self.input_list:
            input_name = os.path.basename(input_path)
            name_wo_ext = os.path.splitext(input_name)[0]
            base_name = name_wo_ext.rsplit('_', 1)[0]  # 提取序列号
            gt_path = find_gt_path(self.gt_dir, base_name)

            if gt_path:
                self.paired_list.append((input_path, gt_path))
            else:
                print(f'[Warning] GT not found for {input_name}, expected base name {base_name}')

        print(f'[TrainDataset] Total paired samples: {len(self.paired_list)}')

    def __len__(self):
        return len(self.paired_list)

    def __getitem__(self, idx):
        input_path, gt_path = self.paired_list[idx]

        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        lr = np.asarray(input_img).astype(np.float32) / 255.0
        gt = np.asarray(gt_img).astype(np.float32) / 255.0

        h, w, _ = lr.shape

        # resize if too small
        if h < self.patch_size or w < self.patch_size:
            scale = self.patch_size / min(h, w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            lr = cv2.resize(lr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = lr.shape[:2]

        if self.crop_patch:
            x = np.random.randint(0, h - self.patch_size + 1)
            y = np.random.randint(0, w - self.patch_size + 1)
            lr = lr[x:x + self.patch_size, y:y + self.patch_size, :]
            gt = gt[x:x + self.patch_size, y:y + self.patch_size, :]

        lr, gt = self.augment(lr, gt)

        lr = T.ToTensor()(lr.copy())
        gt = T.ToTensor()(gt.copy())

        return lr, gt

    def augment(self, lr, gt, hflip=True, rot=True):
        if hflip and random.random() < 0.5:
            lr = lr[:, ::-1, :]
            gt = gt[:, ::-1, :]
        if rot and random.random() < 0.5:
            lr = lr[::-1, :, :]
            gt = gt[::-1, :, :]
        if rot and random.random() < 0.5:
            lr = np.rot90(lr, 2)
            gt = np.rot90(gt, 2)
        return lr, gt



class TestDataset(Dataset):
    def __init__(self, args, mode='test', crop_border=False, crop_patch=False):
        self.dataset_dir = os.path.join(args.dataset_dir, mode)
        self.input_dir = os.path.join(self.dataset_dir, 'input')
        self.gt_dir = os.path.join(self.dataset_dir, 'gt')

        self.input_list = get_image_list(self.input_dir)
        self.paired_list = []

        for input_path in self.input_list:
            input_name = os.path.basename(input_path)
            name_wo_ext = os.path.splitext(input_name)[0]
            base_name = name_wo_ext.rsplit('_', 1)[0]
            gt_path = find_gt_path(self.gt_dir, base_name)

            if gt_path:
                self.paired_list.append((input_path, gt_path))
            else:
                print(f'[Warning] GT not found for {input_name}, expected base name {base_name}')

        self.crop_border = crop_border
        self.crop_patch = crop_patch
        self.patch_size = 512

        print(f'[TestDataset] Total paired samples: {len(self.paired_list)}')

    def __len__(self):
        return len(self.paired_list)

    def __getitem__(self, index):
        input_path, gt_path = self.paired_list[index]
        input_name = os.path.basename(input_path)

        lr_input = Image.open(input_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        lr_t = np.asarray(lr_input).astype(np.float32) / 255.0
        gt_t = np.asarray(gt).astype(np.float32) / 255.0

        lr_t = T.ToTensor()(lr_t)
        gt_t = T.ToTensor()(gt_t)

        if self.crop_border:
            h, w = lr_t.shape[1:]
            new_h = h - h % 32
            new_w = w - w % 32
            lr_t = lr_t[:, :new_h, :new_w]
            gt_t = gt_t[:, :new_h, :new_w]

        if self.crop_patch:
            h, w = lr_t.shape[1:]
            x = np.random.randint(0, h - self.patch_size + 1)
            y = np.random.randint(0, w - self.patch_size + 1)
            lr_t = lr_t[:, x:x+self.patch_size, y:y+self.patch_size]
            gt_t = gt_t[:, x:x+self.patch_size, y:y+self.patch_size]

        return lr_t, gt_t, input_name
