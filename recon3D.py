import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from myUtil import *
from tqdm import tqdm
import wandb
from modules.Unet import UNetModel
from INR_data import VolumeDataset, postprocess_volume, input_mapping, INRInputData, VolumeLatentDataset, preprocess_volume
from core import adjust_learning_rate, evaluate
import matplotlib.pyplot as plt
import imageio
from modules.TIF import TIFModel

# 定义INR模型
exp_name = f"TIF-encoder-lr-decay"
data_path = 'data'
groups = acdc_patients_group(data_path)
group = groups['01.012.OS']
img_sequence, volume_sequence = read_mats_tensor(group)

# 定义训练参数
is_train = True
is_wandb = False if is_train else False
RANDOM_SEED = 42  # any random number
num_epochs = 1000
base_lr = 0.0001
device = 'cuda:3'
model_save_path = 'saved_models'
save_interval = 200
set_seed(RANDOM_SEED)

if is_wandb:
    wandb.login(key="68e06a2142e71f48a9e7352a956bbcdb75675591")
    wandb.init(project="4D_pixel_INR", name=exp_name, config={})

print("超参数准备完毕")
# 拟合图像
# (11, 64, 224, 224)
volume_sequence = img_sequence
min_val, max_val = (volume_sequence).min(), (volume_sequence).max()
volume_sequence = preprocess_volume(volume_sequence)
print('训练数据shape:', volume_sequence.shape)
# 坐标获取工具
inr_input_util = INRInputData(None, (11, 64, 128, 128), device)

unet_config = {'image_size': (128, 128), 'in_channels': 1,
                       'out_channels': 1, 'model_channels': 32, 'dims': 3,
                       'attention_resolutions': [], 'num_res_blocks': 1, 'channel_mult': [1, 2, 4],
                       'num_heads': 8, 'use_scale_shift_norm': True, 'resblock_updown': True,
                       }

model = UNetModel(**unet_config).to(device)
print("模型初始化完成, 开始训练")
if is_train:
    # 加载数据
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    # 训练模型
    step = 0
    temp_lr = -1
    for epoch in range(num_epochs):
        timesteps = np.linspace(0, 10, 11, dtype=np.int32)
        timeIdx = list(range(len(timesteps)))
        random.shuffle(timeIdx)
        labels = volume_sequence.to(device)
        inner_batch_size = 4096
        # 训练模型
        for i in tqdm(timeIdx, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            optimizer.zero_grad()
            # 扩展成图像大小
            t = torch.from_numpy(np.array(timesteps[i])).float().to(device)
            t = t.view(-1, 1)

            # label = labels[i:i+1]
            # 前后两帧合并输入
            # inputs = torch.cat([labels[0:1], labels[10:11]], dim=0)
            inputs = torch.ones_like(labels[0:1].unsqueeze(0))
            label = labels[0:1].unsqueeze(0)
            # coords = np.stack([coord_t, randx, randy, randz], axis=1)
            # coord_list = coords.astype(np.int32).tolist()
            # label = labels[coord_t, randx, randy, randz].to(device)

            # 坐标归一化 [-1, 1], 注意时间维度也归一化了
            # coords = np.stack([coord_t, randx, randy, randz], axis=1).astype(np.float32)
            # normd_coords = inr_input_util.normalize(coords) * 2 - 1
            # coords = torch.from_numpy(normd_coords).float().to(device)

            # 增加一个batch维度
            # coords = coords.unsqueeze(0)
            # label = label.unsqueeze(0)
            # inputs = inputs.unsqueeze(0)

            outputs = model(inputs, timesteps=torch.ones((1,)).to(device), coords=-1)
            loss = criterion(outputs.squeeze(), label.squeeze())

            if is_wandb:
                wandb.log({'loss': loss.item()}, step=step)

            loss.backward()
            optimizer.step()
            step += 1
            if step % save_interval == 0:
                # evaluate(model, inr_input_data, use_wandb=is_wandb, step=step - 1)
                # (1, 1, 64, 224, 224)

                # 预测一帧
                predictions = postprocess_volume(outputs, min_val, max_val)

                slice = predictions.squeeze()[32].detach()

                im_show = slice.cpu().numpy().astype(np.uint8)

                plt.imshow(im_show, cmap='gray')
                plt.show()

                if is_wandb:
                    wandb.log({
                        'inr_pred': [wandb.Image(im_show, caption='inr_pred')],
                        'inr_gt': [wandb.Image(im_gt, caption='inr_gt')]
                        },
                    step=step)

                print(f'保存模型, 第{step}步, Loss: {loss.item():.4f}')
                torch.save(model.state_dict(), f'{model_save_path}/{exp_name}_inr.pth')

            # cur_lr = adjust_learning_rate(optimizer, step, base_lr, interval=1000)
            # if cur_lr != temp_lr:
            #     print(f'调整学习率, 当前学习率: {cur_lr}')
            #     temp_lr = cur_lr

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 保存模型
        torch.save(model.state_dict(), f'{model_save_path}/{exp_name}_inr.pth')

else:
    # 模型评估
    model.load_state_dict(torch.load(f'{model_save_path}/{exp_name}_inr.pth'))
    gif = []
    timesteps = np.linspace(0, 1, 40)
    timeIdx = list(range(len(timesteps)))
    random.shuffle(timeIdx)
    labels = volume_sequence.unsqueeze(1).to(device)
    # 训练模型
    for i in tqdm(timeIdx, desc=f'test', unit='batch'):
        t = torch.from_numpy(np.array(timesteps[i])).float().to(device)
        t = t.view(-1, 1)

        label = labels[i:i + 1]
        # 前后两帧合并输入
        inputs = torch.cat([labels[0:1], labels[10:11]], dim=1)
        # outputs = model(inputs, 0)
        with torch.no_grad():
            outputs = model(inputs, timesteps=torch.ones((1,)).to(device), t_star=t)

        # (1, 1, 64, 224, 224)
        predictions = postprocess_volume(outputs, min_val, max_val)
        gt = postprocess_volume(labels, min_val, max_val)
        slice = predictions[0, 0, 32].detach()
        slice_gt = gt[0, 0, 32].detach()

        im_show = slice.cpu().numpy().astype(np.uint8)
        im_gt = slice_gt.cpu().numpy().astype(np.uint8)
        gif.append(im_show)
        # plt.imshow(im_show, cmap='gray')
        # plt.show()
        # plt.imshow(im_gt, cmap='gray')
        # plt.show()

    # 保存gif
    imageio.mimsave(f'images/{exp_name}_inr.gif', gif, duration=0.5)
