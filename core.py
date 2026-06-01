import torch
import matplotlib.pyplot as plt
import wandb
import numpy as np
from dataset.INR_data import VolumeDataset, postprocess_volume, input_mapping, INRInputData
from my_mds.MyINR import INRModel_v2 as INRModel
import cv2
import os
import shutil
from argparse import ArgumentParser
from myUtil import *
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import cv2 as cv
import albumentations as A
from torchvision import transforms
from natsort import natsorted
from tqdm import tqdm
import imageio

from torchmetrics.image import (PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)
def calculate_nmse(processed_image, original_image):
    sum_diff = torch.sum((original_image - processed_image) ** 2)
    nmse = sum_diff / torch.sum(original_image ** 2)
    # print(torch.sum(original_image ** 2))
    return nmse

SSIM = StructuralSimilarityIndexMeasure(data_range=(-1, 1))
PSNR = PeakSignalNoiseRatio(data_range=(-1, 1))

def predict(model, inputs):
    with torch.no_grad():
        predictions = model(inputs)
    return predictions

def evaluate(model, inr_input_data, use_wandb=False, eval_steps=-1, device=None):
    '''
    训练过程中验证
    :param model:
    :param volume_sequence:
    :param use_wandb:
    :param step:
    :return:
    '''
    test_slice = 42
    time_steps, depth, height, width = inr_input_data.shape
    # device = inr_input_data.device
    B_gauss = inr_input_data.B_gauss
    # test_slice = 3
    if eval_steps == -1:
        # 只预测一帧
        t = 3
        # 尝试预测其中一帧的运动
        res = np.ones((time_steps, height, width)).astype(np.uint8)
        # print(f'预测第{t + 1}帧, slice {test_slice}')
        print(f'预测第{t + 1}帧')

        inputs = inr_input_data[t:t + 1, test_slice:test_slice + 1, :, :]
        inputs = inr_input_data.normalize(inputs)

        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        inputs = input_mapping(inputs, B_gauss.to(device), scale=1.0)

        # de_vo = decode_layer_coord(volume_sequence[t:t + 1, :, :, :].cpu().numpy().astype(np.uint16), y_dim=224)

        predictions = predict(model, inputs)
        predictions = predictions.cpu().numpy().squeeze()
        predictions = postprocess_volume(predictions, inr_input_data.minVal, inr_input_data.maxVal)

        # predictions = predictions * 2 - 1
        if inr_input_data.vae is not None:
            res = res.astype(np.float32)
            res[t, :, :, :] = predictions.reshape(height, width, 4)
        else:
            res[t, :, :] = predictions.reshape(height, width).astype(np.uint16)

        im_show = cv2.cvtColor(res[t, :, :].astype(np.uint8), cv2.COLOR_GRAY2RGB)
        im_gt_show = cv2.cvtColor(inr_input_data.data[t, test_slice, :, :].cpu().numpy().astype(np.uint8),
                                  cv2.COLOR_GRAY2RGB)
        # cv2.imwrite('images/inr_hot.png', hot)
        plt.imshow(im_show)
        plt.show()
        plt.imshow(im_gt_show)
        plt.show()

        im_show = torch.from_numpy(im_show).unsqueeze(0).float() / 127.5 - 1
        im_gt_show = torch.from_numpy(im_gt_show).unsqueeze(0).float() / 127.5 - 1
        im_show = im_show.permute(0, 3, 1, 2)
        im_gt_show = im_gt_show.permute(0, 3, 1, 2)
        ssim = SSIM(im_show, im_gt_show)
        psnr = PSNR(im_show, im_gt_show)
        nmse = calculate_nmse(im_show, im_gt_show)
        print(f"SSIM: {ssim.item()}, PSNR: {psnr.item()}, NMSE: {nmse.item()}")

    else:
        # 预测多个连续时间帧
        # res = np.ones((eval_steps, height, width)).astype(np.uint8)
        time_slices = (0.0, 10.0, 0.25)
        total_step = int((time_slices[1] - time_slices[0]) / time_slices[2])
        inputs = inr_input_data[time_slices, test_slice:test_slice + 1, :, :].astype(np.float64)
        # 尝试预测非整数帧
        inputs = inr_input_data.normalize(inputs)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        inputs = input_mapping(inputs, B_gauss.to(device))
        predictions = predict(model, inputs)
        predictions = predictions.cpu().numpy().squeeze()
        predictions = postprocess_volume(predictions, inr_input_data.minVal, inr_input_data.maxVal)
        # predictions = predictions * 2 - 1
        res = predictions.reshape(total_step, height, width).astype(np.uint16)
        gif = []
        for t in range(total_step):
            im_show = cv2.cvtColor(res[t, :, :].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            # im_gt_show = cv2.cvtColor(inr_input_data.data[t, test_slice, :, :].cpu().numpy().astype(np.uint8),
            #                           cv2.COLOR_GRAY2RGB)
            # plt.imshow(im_show)
            # plt.show()
            #
            # plt.imshow(im_gt_show)
            # plt.show()

            gif.append(im_show)

        imageio.mimsave('images/inr_pred.gif', gif, duration=0.1)

    # de_vo = decode_layer_coord(volume_sequence[t:t + 1, test_slice:test_slice + 1, :, :], y_dim=224)
    # de_res = decode_layer_coord(res[t:t + 1, :, :, :], y_dim=224)
    # latent_tensor = torch.from_numpy(res[t].transpose(2, 0, 1)).unsqueeze(0)
    # val_img = inr_input_data.vae.decode(latent_tensor.to(device))
    # val_img = torch.mean(val_img, dim=1)[0]
    # val_img = (val_img + 1) * 127.5
    # im_show = val_img.cpu().numpy().astype(np.uint8)

    # inr_input_data.vae.decode(
    #     torch.from_numpy(predictions.reshape(height, width, 4).transpose(2, 0, 1)).unsqueeze(0).to(device)).shape
    # diff = np.abs(res[t, :, :] - inr_input_data.data[t, test_slice, :, :].cpu().numpy().astype(np.uint8))
    #
    # diff *= 31
    # diff = diff.astype(np.uint8)
    # hot = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)

    # if use_wandb:
    #     wandb.log({
    #         # 'inr_hot': [wandb.Image(hot, caption='inr_hot')],
    #         'inr_pred': [wandb.Image(im_show, caption='inr_pred')],
    #         'inr_gt': [wandb.Image(im_gt_show, caption='inr_gt')],
    #         # 'hot_sum': float(np.sum(diff))
    #     }, step=step)
    #
    # else:
    #     plt.imshow(hot)
    #     plt.show()


def adjust_learning_rate(optimizer, step, initial_lr, interval=200):
    """学习率衰减"""
    lr = initial_lr * (0.97 ** (step // interval))
    # print(f"学习率调整为{lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        local_path = "/home/Data/maxiao/HuggingFace/models/runwayml/stable-diffusion-v1-5/vae/"
        if os.path.isdir(local_path):
            print(f"Loading VAE from local path: {local_path}")
            self.vae = AutoencoderKL.from_pretrained(local_path)
        else:
            try:
                print("Local VAE not found, loading runwayml/stable-diffusion-v1-5 from Hugging Face...")
                self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
            except Exception as e:
                print("Failed to load VAE from Hugging Face, using dummy identity VAE fallback.", e)
                self.vae = None
        self._scale = 1.0

    @torch.no_grad()
    def encode(self, x):
        if self.vae is None:
            return x
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor * self._scale
        return latents

    @torch.no_grad()
    def decode(self, latents):
        if self.vae is None:
            return latents
        latents /= self.vae.config.scaling_factor
        latents = latents / self._scale
        rec = self.vae.decode(latents, return_dict=False)[0]
        return rec

def inr_dataset_group(data_path):
    # HyperINR-02-015_OD_S2T.mat_inr.pth_inr_12_9.mat
    groups = {}
    # 同一个病人的同一个眼睛为一组
    for file in os.listdir(data_path):
        group_name = file.split('.')[0]
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(file)
    return groups


if __name__ == '__main__':
    model = VAE()
    model.eval()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data_path = 'data'
    num_epochs = 10
    groups = acdc_patients_group(data_path)
    group = groups['01.012.OS']
    img_sequence, volume_sequence = read_mats_tensor(group)

    with torch.no_grad():
        x = img_sequence[4, 3].numpy()
        transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize((0.5,), (0.5,))
        ])
        x = transform(image=x)['image']
        x = transforms.ToTensor()(x)

        x = x.repeat(1, 3, 1, 1)
        x = x.to(device)
        z = model.encode(x)
        print(z.shape)
        # np.save(os.path.join(cube_dir, f'{cube_name}_{i}.npy'), z.cpu().numpy())
        xrec = model.decode(z)
        xrec = torch.mean(xrec, dim=1, keepdim=True)

        plt.imshow(x.cpu().numpy()[0, 0])
        plt.show()

        plt.imshow(xrec.cpu().numpy()[0, 0])
        plt.show()