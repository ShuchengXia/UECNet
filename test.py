import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import random
import numpy as np
import argparse
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
# from data.BAID_data import TrainDataset, TestDataset
# from data.LCDP_data import TrainDataset, TestDataset
# from data.MSEC_data import TrainDataset, TestDataset
# from utils.utils_data import TrainDataset, TestDataset
from data.SICE_DE_data import TrainDataset, TestDataset
from piqa import PSNR, SSIM
import pyiqa
import torchvision
from datetime import datetime
from tqdm import tqdm
import time
from model.UECNet import UECNet


def get_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def time_synchronized(use_cpu=False):
    torch.cuda.synchronize(
    ) if torch.cuda.is_available() and not use_cpu else None
    return time.time()


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    datasets = ['Backlit300', 'BAID-resize', 'LCDPNet', 'samples']
    dataset = datasets[2]
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset-dir', type=str, default=f'/ext_ssd/xsc_datasets/{dataset}/')
    # parser.add_argument('--dataset-dir', type=str, default=f'./samples/')
    parser.add_argument('--dataset-dir', type=str, default='./datasets/LCDP/', help='dataset dir')
    parser.add_argument('--result-dir', type=str, default='./result/LCDP/')
    args = parser.parse_args()


    save_file = True
    need_GT = True
    if dataset == "Backlit300" or dataset == "samples":
        need_GT = False
    valset = TestDataset(args, mode='test', crop_border=True)
    valloader = DataLoader(dataset=valset, batch_size=1)

    net = UECNet(channel=32).cuda()

    # 加载训练好的参数
    path = './ckpt/sice_de.pth'
    print(path)
    net.load_state_dict(torch.load(path), strict=False)

    psnr = PSNR()
    ssim = SSIM().cuda()
    musiq = pyiqa.create_metric('musiq', device=torch.device("cuda"))

    net.eval()
    with torch.no_grad():
        sum_psnr = 0.0
        sum_ssim = 0.0
        total_lpips = 0.0
        total_musiq = 0.0
        total_niqe = 0.0

        if need_GT:
            with tqdm(total=len(valloader), desc='Processing', leave=True, ncols=100, unit='it', unit_scale=True) as pbar:
                # save_path = f'./exp_results/{net_name}/{basic_unit}/{exp_name}/{dataset}/'
                save_path = args.result_dir
                os.makedirs(save_path, exist_ok=True)
                for val_iteration, val_data in enumerate(valloader):
                    val_input, val_label, filename = val_data[0].cuda(), val_data[1].cuda(), val_data[2]
                    file_name = filename[0]
                    val_output = net(val_input)

                    out_psnr = psnr(torch.clamp(val_output, 0, 1), val_label)
                    out_ssim = ssim(torch.clamp(val_output, 0, 1), val_label)
                    out_musiq = musiq(torch.clamp(val_output, 0, 1))

                    total_musiq += out_musiq.item()

                    sum_psnr += out_psnr
                    sum_ssim += out_ssim

                    if save_file:
                        # print(f"file_name: {file_name}, type: {type(file_name)}")
                        name, ext = os.path.splitext(file_name)
                        torchvision.utils.save_image(val_output, os.path.join(save_path, file_name))
                    pbar.update()

            val_psnr = sum_psnr / len(valloader)
            val_ssim = sum_ssim / len(valloader)
            musiq_mean = total_musiq / len(valloader)
            print(len(valloader))

            print("\n")
            print(f'[PSNR/SSIM]: {val_psnr:.4f} / {val_ssim:.4f}')
            print(f'[MUSIQ]: {musiq_mean:.4f}')
            print(f'[ * ] End Time: {get_now()}\n')

        else:
            # save_path = f'./exp_results/{net_name}/{basic_unit}/{exp_name}/{dataset}/'
            save_path = f'./samples/output/'
            os.makedirs(save_path, exist_ok=True)
            with tqdm(total=len(valloader), desc='Processing', leave=True, ncols=100, unit='it', unit_scale=True) as pbar:
                for val_iteration, val_data in enumerate(valloader):
                    val_input, val_attn_map, file_name = val_data[0].cuda(), val_data[1].cuda(), val_data[2]
                    file_name = file_name[0]
                    val_output = net(val_input, val_attn_map)

                    out_musiq = musiq(torch.clamp(val_output, 0, 1))
                    total_musiq += out_musiq.item()

                    if save_file:
                        torchvision.utils.save_image(val_output, os.path.join(save_path, file_name))
                    pbar.update()

            musiq_mean = total_musiq / len(valloader)
            # print(f'\nDone!')
            print(f'[MUSIQ]: {musiq_mean:.4f}')
            print(f'[ * ] End Time: {get_now()}\n')

