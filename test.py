import os
import sys

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
from data.SICE-DE_dataset import TrainDataset, TestDataset
from utils.utils import cal_psnr, cal_ssim
from piqa import PSNR, SSIM  # , LPIPS
import pyiqa
import torchvision
from datetime import datetime
from tqdm import tqdm
import time
from model.LightRestore_v6 import LightRestore


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
    # parser.add_argument('--dataset-dir', type=str, default=f'/ext_ssd/datasets/{dataset}/')
    # parser.add_argument('--dataset-dir', type=str, default=f'./samples/')
    parser.add_argument('--dataset-dir', type=str, default='/ext_ssd/datasets/LCDPNet/', help='dataset dir')
    parser.add_argument('--result-dir', type=str, default='/test/')
    args = parser.parse_args()


    save_file = True
    need_GT = True
    if dataset == "Backlit300" or dataset == "samples":
        need_GT = False
    valset = TestDataset(args, mode='test', crop_border=True)
    valloader = DataLoader(dataset=valset, batch_size=1)

    net = LightRestore(channel=32).cuda()

    # FLOPs & Running Time
    """
    x = torch.randn((1, 3, 512, 512)).cuda()
    y = torch.randn((1, 1, 512, 512)).cuda()
    print(clever_format(profile(net, (x, y))), "%.5")

    t1 = 0
    for i in range(1000):
        t = time_synchronized()
        z = net(x, y)  # inference and training outputs
        t1 += time_synchronized() - t

    t1 = t1 / 1000 * 1E3  # speeds per image
    print(f'[ * ] Running Time = %.4fms' % t1)
    sys.exit()
    """

    # 加载训练好的参数
    path = '/home/xsc/LightRestore/result/LCDP/0701_2301/best_psnr.pth'
    print(path)
    net.load_state_dict(torch.load(path), strict=False)

    psnr = PSNR()
    ssim = SSIM().cuda()
    # lpips = LPIPS().cuda()
    lpips = pyiqa.create_metric('lpips', device=torch.device("cuda"))
    niqe = pyiqa.create_metric('niqe', device=torch.device("cuda"))
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

                    # val_input = T.Resize((512, 512))(val_input)
                    # val_attn_map = T.Resize((512, 512))(val_attn_map)
                    # val_label = T.Resize((512, 512))(val_label)

                    val_output, encoder_out, middle_out = net(val_input)

                    # """
                    # out_psnr = cal_psnr(torch.clamp(val_output, 0, 1), val_label)
                    # out_ssim = cal_ssim(torch.clamp(val_output, 0, 1), val_label)
                    # """
                    out_psnr = psnr(torch.clamp(val_output, 0, 1), val_label)
                    out_ssim = ssim(torch.clamp(val_output, 0, 1), val_label)
                    out_lpips = lpips(torch.clamp(val_output, 0, 1), val_label)
                    out_musiq = musiq(torch.clamp(val_output, 0, 1))
                    out_niqe = niqe(torch.clamp(val_output, 0, 1))

                    encoder_out = torch.sigmoid(encoder_out)  # [B, 3, H, W]
                    print(encoder_out.shape)
                    middle_out = torch.sigmoid(middle_out)  # [B, 3, H, W]

                    if out_lpips != out_lpips:
                        out_lpips = 0
                        print(f"LPIPS value of {file_name} is NaN.")
                    total_lpips += out_lpips
                    total_musiq += out_musiq.item()
                    total_niqe += out_niqe

                    sum_psnr += out_psnr
                    sum_ssim += out_ssim

                    if save_file:
                        # print(f"file_name: {file_name}, type: {type(file_name)}")
                        name, ext = os.path.splitext(file_name)
                        torchvision.utils.save_image(val_output, os.path.join(save_path, file_name))
                        torchvision.utils.save_image(encoder_out, os.path.join(save_path, f"{name}_encoder{ext}"))
                        torchvision.utils.save_image(middle_out, os.path.join(save_path, f"{name}_middle{ext}"))

                    pbar.update()

            val_psnr = sum_psnr / len(valloader)
            val_ssim = sum_ssim / len(valloader)
            lpips_mean = total_lpips / len(valloader)
            musiq_mean = total_musiq / len(valloader)
            niqe_mean = total_niqe / len(valloader)
            print(len(valloader))

            print("\n")
            print(f'[PSNR/SSIM]: {val_psnr:.4f} / {val_ssim:.4f}')
            print(f'[LPIPS]: {lpips_mean:.4f}')
            print(f'[NIQE]: {niqe_mean:.4f}')
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

                    out_niqe = niqe(torch.clamp(val_output, 0, 1))
                    out_musiq = musiq(torch.clamp(val_output, 0, 1))
                    total_niqe += out_niqe
                    total_musiq += out_musiq.item()

                    if save_file:
                        torchvision.utils.save_image(val_output, os.path.join(save_path, file_name))
                    pbar.update()

            niqe_mean = total_niqe / len(valloader)
            musiq_mean = total_musiq / len(valloader)
            # print(f'\nDone!')
            print(f'[NIQE]: {niqe_mean:.4f}')
            print(f'[MUSIQ]: {musiq_mean:.4f}')
            print(f'[ * ] End Time: {get_now()}\n')


