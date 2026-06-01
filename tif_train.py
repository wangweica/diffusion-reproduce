import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from myUtil import *
from tqdm import tqdm
import wandb
from dataset.INR_data import VolumeDataset, postprocess_volume, input_mapping, INRInputData, VolumeLatentDataset, preprocess_volume
from core import adjust_learning_rate, evaluate
import matplotlib.pyplot as plt
import imageio
from my_mds.MyINR import INRModel_v2 as INRModel
from my_mds.INRs import HyperINRModel
from dataset.TIF_Dataset import TIF_Dataset

# 定义INR模型
exp_name = f"HyperINR"
device = 'cuda:3'
data_path = 'data'
model_save_path = 'saved_models'

is_train = True
is_wandb = False if is_train else False

RANDOM_SEED = 42  # any random number
num_epochs = 1000
base_lr = 0.00003
save_interval = 100
inner_batch_size = 12288
set_seed(RANDOM_SEED)

if is_wandb:
    wandb.login(key="68e06a2142e71f48a9e7352a956bbcdb75675591")
    wandb.init(project="4D_pixel_INR", name=exp_name, config={})

dataset = TIF_Dataset(data_path, device, inner_batch_size=inner_batch_size)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
# model = TIFModel(in_channels=2, out_channels=1).to(device)
# channel_list = [4]
# network_width = 256
# network_depth = 6
# for depth in range(network_depth):
#     channel_list.append(network_width)
#     if depth == network_depth - 1:
#         channel_list.append(1)
# model = INRModel(channel_list, skip_to=[]).to(device)
model = HyperINRModel(input_dim=2, hidden_dim=256, target_shapes=[(256, 256)] * 6, rank=64).to(device)
model.train()

if is_train:
    print("模型初始化完成, 开始训练")
    # 加载数据
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    # 训练模型
    step = 0
    temp_lr = -1
    # all_coords = inr_input_util[1:2, :, :, :]
    for epoch in range(num_epochs):
        # 训练模型
        for coords, cond, label in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            optimizer.zero_grad()
            coords, cond, label = dataset.toDevice([coords, cond, label], device)
            coords = coords.squeeze()
            cond = cond
            label = label.squeeze()
            outputs = model(y=cond, q=coords)
            print("train output:", outputs.mean().item(), outputs.std().item())
            loss = criterion(outputs.squeeze(), label.squeeze())

            if is_wandb:
                wandb.log({'loss': loss.item()}, step=step)

            loss.backward()
            optimizer.step()
            step += 1
            if step % save_interval == 0:
                # 预测一帧
                coords = dataset.get_slice_coords([(0, 1), (42, 43), (0, 128), (0, 128)])
                coords = coords
                coords = coords.to(device)
                with torch.no_grad():
                    model.eval()
                    outputs = model(y=cond, q=coords)
                    outputs = outputs.reshape(128, 128)
                    print("eval output:", outputs.mean(), outputs.std())
                    model.train()
                predictions = dataset.postprocess_volume(outputs)
                gt = dataset.postprocess_volume(dataset.volume_sequence[0:1, 42:43, :, :])

                slice = predictions.squeeze().detach()
                slice_gt = gt.squeeze().detach()

                im_show = slice.cpu().numpy().astype(np.uint8)
                # im_gt = slice_gt.cpu().numpy().astype(np.uint8)

                plt.imshow(im_show, cmap='gray')
                plt.show()
                # plt.imshow(im_gt, cmap='gray')
                # plt.show()

                if is_wandb:
                    wandb.log({
                        'inr_pred': [wandb.Image(im_show, caption='inr_pred')],
                        # 'inr_gt': [wandb.Image(im_gt, caption='inr_gt')]
                        },
                    step=step)

                print(f'保存模型, 第{step}步, Loss: {loss.item():.5f}')
                torch.save(model.state_dict(), f'{model_save_path}/{exp_name}_inr.pth')

            cur_lr = adjust_learning_rate(optimizer, step, base_lr, interval=1000)
            if cur_lr != temp_lr:
                print(f'调整学习率, 当前学习率: {cur_lr}')
                temp_lr = cur_lr

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 保存模型
        torch.save(model.state_dict(), f'{model_save_path}/{exp_name}_inr.pth')

else:
    # 模型评估
    model.load_state_dict(torch.load(f'{model_save_path}/{exp_name}_inr.pth'))
    gif = []
    timesteps = np.linspace(0, 10, 11, dtype=np.int64)
    timeIdx = list(range(len(timesteps)))
    labels = dataset.volume_sequence.unsqueeze(1).to(device)

    for i in tqdm(timeIdx, desc=f'test', unit='batch'):
        t = torch.from_numpy(np.array(timesteps[i])).float().to(device)
        t = t.view(-1, 1)

        # 前后两帧合并输入
        cond = torch.cat([labels[0:1], labels[10:11]], dim=1)
        # coords = dataset.inr_input_util[(timesteps[i], timesteps[i]+1, 301), (42, 43, 43), :, :]
        # coords = dataset.inr_input_util.normalize(coords)
        # coords = torch.from_numpy(coords).float()
        coords = dataset.get_slice_coords([(timesteps[i], timesteps[i]+1), (42, 43), (0, 128), (0, 128)])
        coords, cond = dataset.toDevice([coords, cond], device)
        with torch.no_grad():
            model.eval()
            outputs = model(y=cond, q=coords)
            outputs = outputs.reshape(128, 128)
            print("eval output:", outputs.mean(), outputs.std())

        # (1, 1, 64, 224, 224)
        predictions = postprocess_volume(outputs, dataset.min_val, dataset.max_val)
        gt = postprocess_volume(labels, dataset.min_val, dataset.max_val)
        slice = predictions.detach()
        slice_gt = gt.detach()

        im_show = slice.cpu().numpy().astype(np.uint8)
        im_gt = slice_gt.cpu().numpy().astype(np.uint8)
        gif.append(im_show)
        # plt.imshow(im_show, cmap='gray')
        # plt.show()
        # plt.imshow(im_gt, cmap='gray')
        # plt.show()

    # 保存gif
    imageio.mimsave(f'images/{exp_name}_inr.gif', gif, duration=0.5)
