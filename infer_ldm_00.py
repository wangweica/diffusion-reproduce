from train_ldm import LDM, get_parser
import torch
import numpy as np
import scipy.io as sio
import os
import argparse
from tqdm import tqdm
from core import *
from myUtil import run_histogram_match
from torchmetrics.image import (PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)
import matplotlib.pyplot as plt
import albumentations as A
from torchvision import transforms

def calculate_nmse(processed_image, original_image):
    sum_diff = torch.sum((original_image - processed_image) ** 2)
    nmse = sum_diff / torch.sum(original_image ** 2)
    # print(torch.sum(original_image ** 2))
    return nmse

input_mat = sio.loadmat("/home/Data/zhangxiao/inr/out/paper_show/HyperINR-02-043_OS_S2T.mat_inr.pth_inr_2.mat")
# input_mat = sio.loadmat("/home/Data/zhangxiao/inr/out/inr2img_dataset_acdc_test/patient092_inr_inr_5.mat")
# S_mat = sio.loadmat("/home/Data/zhangxiao/inr/out/inr2img_dataset_acdc_test/patient092_inr_inr_0.mat")
# T_mat = sio.loadmat("/home/Data/zhangxiao/inr/out/inr2img_dataset_acdc_test/patient092_inr_inr_14.mat")
model_path = "/home/Data/zhangxiao/inr/out/inr_refine_log/vaeldm_2d_batch_cond_fulltuning_num_head_4/lightning_logs/version_19/checkpoints/last.ckpt"
# model_path = "/home/Data/zhangxiao/inr/out/inr_refine_log/acdc_vaeldm_2d_batch_cond_fulltuning_num_head_4/lightning_logs/version_2/checkpoints/model-epoch=419.ckpt"

device = 'cuda:0'
cube_gen = input_mat['lr']
cube_gt = input_mat['hr']

opts = get_parser().parse_args()
model = LDM(opts)
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['state_dict'])
model.eval()
model.first_stage_model = model.first_stage_model.to(device)
model = model.to(device)

B, H, W = cube_gen.shape
print(cube_gen.shape)
transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize((0.5,), (0.5,))
])
SSIM = StructuralSimilarityIndexMeasure(data_range=255)
PSNR = PeakSignalNoiseRatio(data_range=255)

def infer_one_cube(img_sequence):
    S, H, W = img_sequence.shape

    cube_res = []
    for s in range(S):
        x = img_sequence[s]
        x = transform(image=x)['image']
        x = transforms.ToTensor()(x)

        x = x.repeat(1, 3, 1, 1)
        x = x.to(device)

        with torch.no_grad():
            z = model.first_stage_model.encode(x)
            cube_res.append(z.squeeze().cpu())
    return torch.stack(cube_res, dim=0)

cube_gen_z = infer_one_cube(cube_gen[0:8])
cube_gt_z = infer_one_cube(cube_gt)

B = cube_gen_z.shape[0]
# 循环采样
with torch.no_grad():
    # se_frame = torch.cat([cube_gt_z[0:1, :, :, :], cube_gt_z[-1:, :, :, :]], dim=1)
    # se_frame = se_frame.repeat(B, 1, 1, 1)
    # c = torch.cat([cube_gen_z, se_frame], dim=1)
    # x = model.sample(c=c, batch_size=B, return_intermediates=False, clip_denoised=False)
    # # 解码
    # x = model.first_stage_model.decode(x)
    # x = x.cpu().numpy()
    psnr_total = 0
    ssim_total = 0
    nmse_total = 0

    # 创建保存图片的目录
    save_dir = "img_out"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(B):
        # img_gen = x[i].transpose(1, 2, 0)[:, :, 0] * 0.5 + 0.5
        # img_gen = img_gen.clip(0, 1) * 255
        # img_gen = img_gen.astype(np.uint8)
        # img_gen = run_histogram_match(img_gen, img_gt)

        img_gt = transform(image=cube_gt[i])['image']
        img_gt = img_gt * 0.5 + 0.5
        img_gt = img_gt.clip(0, 1) * 255
        img_gt = img_gt.astype(np.uint8)
        

        # 保存生成的图片
        # plt.imshow(img_gen, cmap='gray')
        # plt.savefig(os.path.join(save_dir, f'generated_image_{i}.png'))
        # plt.close()

        # 保存真实的图片
        plt.imshow(img_gt, cmap='gray')
        plt.savefig(os.path.join(save_dir, f'ground_truth_image_{i}.png'))
        plt.close()

    #     img_gen = torch.from_numpy(img_gen).unsqueeze(0).unsqueeze(0).float()
    #     img_gt = torch.from_numpy(img_gt).unsqueeze(0).unsqueeze(0).float()
    #     psnr = PSNR(img_gen, img_gt)
    #     ssim = SSIM(img_gen, img_gt)
    #     nmse = calculate_nmse(img_gen, img_gt)
    #     psnr_total += psnr
    #     ssim_total += ssim
    #     nmse_total += nmse
    #     print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, NMSE: {nmse:.4f}")
    # print(f"Average PSNR: {psnr_total/B:.4f}, Average SSIM: {ssim_total/B:.4f}, Average NMSE: {nmse_total/B:.4f}")
    print("Done")