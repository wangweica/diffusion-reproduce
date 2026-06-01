import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from myUtil import *
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
from dataset.INR_data import VolumeDataset, postprocess_volume, input_mapping, INRInputData, VolumeLatentDataset, preprocess_volume
from core import adjust_learning_rate, evaluate
import matplotlib.pyplot as plt
import imageio
from my_mds.MyINR import INRModel_v2 as INRModel
from my_mds.INRs import HyperINRModel
from dataset.TIF_Dataset import TIF_Dataset

from torchmetrics.image import (PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)
def calculate_nmse(processed_image, original_image):
    sum_diff = torch.sum((original_image - processed_image) ** 2)
    nmse = sum_diff / torch.sum(original_image ** 2)
    # print(torch.sum(original_image ** 2))
    return nmse

SSIM = StructuralSimilarityIndexMeasure(data_range=(-1, 1))
PSNR = PeakSignalNoiseRatio(data_range=(-1, 1))


# 定义INR模型
# exp_name = f"HyperINR"
# device = 'cuda:3'
# data_path = 'data'


is_wandb = False

def train(exp_name, device, img, model_save_path = "D:/studio/oct_gen/models/all_inr"):
    RANDOM_SEED = 42  # any random number
    num_epochs = 20
    base_lr = 1e-5
    save_interval = 500
    inner_batch_size = 16384
    set_seed(RANDOM_SEED)
    os.makedirs(model_save_path, exist_ok=True)

    if is_wandb:
        wandb.login(key="68e06a2142e71f48a9e7352a956bbcdb75675591")
        wandb.init(project="4D_pixel_INR", name=exp_name, config={})

    dataset = TIF_Dataset(None, device, inner_batch_size=inner_batch_size, img=img)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = HyperINRModel(input_dim=2, hidden_dim=256, target_shapes=[(-1, -1)] * 6, rank=64).to(device)
    model.train()

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(os.path.join(model_save_path, 'imgs'), exist_ok=True)

    print("模型初始化完成, 开始训练")
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    step = 0

    for epoch in range(num_epochs):
        for batch_idx, (coords, cond, label) in enumerate(tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')):
            optimizer.zero_grad()
            coords, cond, label = dataset.toDevice([coords, cond, label], device)
            coords = coords.squeeze()
            label = label.squeeze()
            outputs = model(y=cond, q=coords)
            loss = criterion(outputs.squeeze(), label.squeeze())

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"训练过程中出现非法 loss={loss.item()}，文件={exp_name}, epoch={epoch+1}, batch={batch_idx}，可能输入或模型参数出现 NaN/Inf"
                )

            if is_wandb:
                wandb.log({'loss': loss.item()}, step=step)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1

        # 训练一个 epoch 后进行一次简单评估并保存模型
        with torch.no_grad():
            coords_eval = dataset.get_slice_coords([(0, 1), (15, 16), (0, 128), (0, 128)])
            coords_eval = coords_eval.to(device)
            model.eval()
            outputs = model(y=cond, q=coords_eval)
            outputs = outputs.reshape(128, 128)
            print("eval output:", outputs.mean(), outputs.std())
            model.train()

            predictions = dataset.postprocess_volume(outputs)
            gt = dataset.postprocess_volume(dataset.volume_sequence[0:1, 15:16, 0:128, 0:128])
            slice_pred = predictions.squeeze().detach()
            slice_gt = gt.squeeze().detach()
            im_show = slice_pred.cpu().numpy().astype(np.uint8)
            im_gt = slice_gt.cpu().numpy().astype(np.uint8)
            plt.imsave(os.path.join(model_save_path, 'imgs', f'{exp_name}_inr_{step}.png'), im_show, cmap='gray')

            im_show = torch.from_numpy(im_show).unsqueeze(0).float() / 127.5 - 1
            im_gt_show = torch.from_numpy(im_gt).unsqueeze(0).float() / 127.5 - 1
            im_show = im_show.unsqueeze(0)
            im_gt_show = im_gt_show.unsqueeze(0)
            if im_show.shape == im_gt_show.shape:
                ssim = SSIM(im_show, im_gt_show)
                psnr = PSNR(im_show, im_gt_show)
                nmse = calculate_nmse(im_show, im_gt_show)
                print(f"SSIM: {ssim.item()}, PSNR: {psnr.item()}, NMSE: {nmse.item()}")
            else:
                print(f"跳过评估指标：预测 {im_show.shape} 与 GT {im_gt_show.shape} 形状不匹配")
            print(f"SSIM: {ssim.item()}, PSNR: {psnr.item()}, NMSE: {nmse.item()}")

            if is_wandb:
                wandb.log({'inr_pred': [wandb.Image(im_show, caption='inr_pred')]}, step=step)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), os.path.join(model_save_path, f'{exp_name}_inr.pth'))
