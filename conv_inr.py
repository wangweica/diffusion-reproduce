import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from myUtil import *
from tqdm import tqdm
import wandb
from MyINR import TimeToImageINR, predict
from Unet import UNetModel
from INR_data import VolumeDataset, postprocess_volume, input_mapping, INRInputData, VolumeLatentDataset, preprocess_volume
# import signal
from core import adjust_learning_rate, evaluate, VAE

# 定义INR模型
is_train = True
is_wandb = True if is_train else False
RANDOM_SEED = 42  # any random number
set_seed(RANDOM_SEED)

num_block = 18
lr = 0.0001
img_size = [64, 64, 64]
exp_name = f"INR_width_conv{18}_bs4096_lr-decay"
if is_wandb:
    wandb.login(key="68e06a2142e71f48a9e7352a956bbcdb75675591")
    wandb.init(project="4D_pixel_INR", name=exp_name, config={})

data_path = 'data'
num_epochs = 100
groups = acdc_patients_group(data_path)
group = groups['01.012.OS']
# img_sequence, volume_sequence = read_mats_tensor(group)
print("超参数准备完毕")
# 改为拟合图像
# (11, 64, 4, 64, 64)
volume_sequence = np.load("latent_data/01.012.OS.npy")
print("数据加载完毕")
volume_sequence = torch.from_numpy(volume_sequence)
min_val, max_val = (volume_sequence).min(), (volume_sequence).max()
volume_sequence = preprocess_volume(volume_sequence)


device = 'cuda:1'
model_save_path = 'models'
save_interval = 40


print('训练数据shape:', volume_sequence.shape)
inr_input_data = INRInputData(volume_sequence, None, device, None, max_val.item(), min_val.item())

# VAE用于评估
vae = VAE()
vae.eval()
vae.to(device)
inr_input_data.vae = vae

# model = UNet(in_channel=9, out_channel=4, inner_channel=32,
#              channel_mults=(1, 2, 2, 4, 4, 8), attn_res=[8, 16],
#              with_time_emb=False, image_size=[64, 64],
#              res_blocks=3).to(device)

unet_config = {'image_size': (64, 64), 'in_channels': 9,
                       'out_channels': 4, 'model_channels': 192,
                       'attention_resolutions': [2, 4, 8], 'num_res_blocks': 3, 'channel_mult': [1, 2, 4, 4],
                       'num_heads': 8, 'use_scale_shift_norm': True, 'resblock_updown': True,
                       }

model = UNetModel(**unet_config).to(device)
print("模型初始化完成, 开始训练")
if is_train:
    # 加载数据
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练模型
    step = 0
    for epoch in range(num_epochs):
        timesteps = np.linspace(0, 1, 11)
        timeIdx = list(range(len(timesteps)))
        random.shuffle(timeIdx)

        for i in tqdm(timeIdx, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            optimizer.zero_grad()
            # 扩展成图像大小
            t = torch.from_numpy(np.array(timesteps[i])).float().to(device)
            t = t.view(-1, 1, 1, 1, 1).expand(-1, 1, img_size[0], img_size[1], img_size[2])
            labels = volume_sequence.permute(0, 2, 1, 3, 4).to(device)
            label = labels[i:i+1, :, 32]
            t = t[:, :, 32]
            # 前后两帧合并输入
            inputs = torch.cat([labels[0:1, :, 32], labels[10:11, :, 32], t], dim=1)
            # outputs = model(inputs, 0)
            outputs = model(inputs, timesteps=torch.ones((1,)).to(device))
            loss = criterion(outputs, label)

            if is_wandb:
                wandb.log({'loss': loss.item()}, step=step)

            loss.backward()
            optimizer.step()
            step += 1
            if step % save_interval == 0:
                # evaluate(model, inr_input_data, use_wandb=is_wandb, step=step - 1)
                # (1, 4, 64, 64, 64)
                predictions = postprocess_volume(outputs, inr_input_data.minVal, inr_input_data.maxVal)
                gt = postprocess_volume(labels, inr_input_data.minVal, inr_input_data.maxVal)
                # slice = predictions[:, :, 6, ...]
                # slice_gt = gt[:, :, 6, ...]
                slice = predictions[:, :, ...]
                slice_gt = gt[:, :, 32, ...]
                val_img = inr_input_data.vae.decode(slice)
                val_img = torch.mean(val_img, dim=1)[0]
                val_img = (val_img + 1) * 127.5

                val_gt = inr_input_data.vae.decode(slice_gt)
                val_gt = torch.mean(val_gt, dim=1)[0]
                val_gt = (val_gt + 1) * 127.5

                im_gt = val_gt.cpu().numpy().astype(np.uint8)
                im_show = val_img.cpu().numpy().astype(np.uint8)
                if is_wandb:
                    wandb.log({
                        'inr_pred': [wandb.Image(im_show, caption='inr_pred')],
                        'inr_gt': [wandb.Image(im_gt, caption='inr_gt')]

                    }, step=step)

                print(f'保存模型, 第{step}步, Loss: {loss.item():.4f}')
                torch.save(model.state_dict(), f'{model_save_path}/{exp_name}_inr.pth')

            # adjust_learning_rate(optimizer, step, lr)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 保存模型
        torch.save(model.state_dict(), f'{model_save_path}/{exp_name}_inr.pth')

else:
    # 模型评估
    model.load_state_dict(torch.load(f'{model_save_path}/{exp_name}_inr.pth'))

    evaluate(model, inr_input_data, wandb=False)

    # # 保存结果
    # print('保存结果在inr_predictions.mat中')
    # sio.savemat('inr_predictions.mat', {'data': res})