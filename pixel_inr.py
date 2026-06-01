import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from myUtil import *
from tqdm import tqdm
import wandb
from MyINR import INRModel_v2 as INRModel, predict
from INR_data import VolumeDataset, postprocess_volume, input_mapping, INRInputData

# import signal
from core import adjust_learning_rate, evaluate

# 定义INR模型
is_train = True
is_wandb = False if is_train else False
RANDOM_SEED = 42  # any random number
set_seed(RANDOM_SEED)

network_width = 256
network_depth = 12
lr = 0.001
exp_name = f"INR_width{network_width}_depth{network_depth}_bs4096_lr-decay"
if is_wandb:
    wandb.login(key="68e06a2142e71f48a9e7352a956bbcdb75675591")
    wandb.init(project="4D_INR", name=exp_name, config={})

data_path = 'data'
num_epochs = 10
groups = acdc_patients_group(data_path)
group = groups['01.012.OS']
img_sequence, volume_sequence = read_mats_tensor(group)

# 改为拟合图像
volume_sequence = img_sequence
# encoded_layer_volume = sio.loadmat(os.path.join(data_path, '01.012.OS.mat'))['data']
# volume_sequence = torch.from_numpy(encoded_layer_volume).float()

device = 'cuda:3'
model_save_path = 'saved_models'
save_interval = 100
if is_train:
    B_gauss = torch.randn((network_width // 2, 4))
    torch.save(B_gauss, 'B_gauss.pth')
else:
    B_gauss = torch.load('B_gauss.pth')

min_val, max_val = (volume_sequence).min(), (volume_sequence).max()
print('训练数据shape:', volume_sequence.shape)
inr_input_data = INRInputData(volume_sequence, volume_sequence.shape, device, B_gauss, max_val.item(), min_val.item())
# 初始化模型、损失函数和优化器
channel_list = []
for depth in range(network_depth):
    channel_list.append(network_width)
    if depth == network_depth - 1:
        channel_list.append(1)

model = INRModel(channel_list, skip_to=[network_depth // 2]).to(device)

if is_train:
    # 加载数据
    dataset = VolumeDataset(volume_sequence, B_gauss=B_gauss, inr_data=inr_input_data)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练模型
    step = 0
    for epoch in range(num_epochs):
        for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            step += 1
            if step % save_interval == 0:
                if is_wandb:
                    wandb.log({'loss': loss.item()}, step=step)

                evaluate(model, inr_input_data, use_wandb=is_wandb)
                print(f'保存模型, 第{step}步, Loss: {loss.item():.4f}')
                torch.save(model.state_dict(), f'{model_save_path}/{exp_name}_inr.pth')

            adjust_learning_rate(optimizer, step, lr)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 保存模型
        torch.save(model.state_dict(), f'{model_save_path}/{exp_name}_inr.pth')

else:
    # 模型评估
    model.load_state_dict(torch.load(f'{model_save_path}/{exp_name}_inr.pth'))

    with torch.no_grad():
        evaluate(model, inr_input_data, use_wandb=False, eval_steps=5)

    # # 保存结果
    # print('保存结果在inr_predictions.mat中')
    # sio.savemat('inr_predictions.mat', {'data': res})