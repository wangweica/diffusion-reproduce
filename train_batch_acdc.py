import os
import scipy.io as sio
import numpy as np
from tif_train_func import train
from natsort import natsorted
from myUtil import *
data_path = "/home/Data/zhangxiao/datasets/acdc_dataset/mats-inter/train"
groups = {}
for file in os.listdir(data_path):
    if file.endswith(".mat"):
        name = file.replace(".mat", "")
        pname, inter = name.split("_inter[")
        inter = float(inter.split("]")[0])
        if pname not in groups:
            groups[pname] = []
        groups[pname].append({'name': name, 'path': os.path.join(data_path, file), 'inter': inter})


device2 = torch.device("cuda:1")
idx = 1
for pname, info_list in groups.items():
    if idx % 2 != 0:
        device2 = torch.device("cuda:1")
        idx += 1

    else:
        device2 = torch.device("cuda:1")
        idx += 1
        continue

    info_list = natsorted(info_list, key=lambda x: x['inter'])
    fourD_data = []
    for i, info in enumerate(info_list):
        mat = sio.loadmat(info['path'])['inter_image']
        fourD_data.append(mat.transpose(2, 0, 1))
    fourD_data = np.stack(fourD_data, axis=0)
    # 归一化
    fourD_data = (fourD_data - fourD_data.min()) / (fourD_data.max() - fourD_data.min())
    fourD_data *= 255
    print(pname, "shape: ", fourD_data.shape)
    train(pname, device2, fourD_data, model_save_path="/home/Data/zhangxiao/inr/models/all_inr_acdc/")

        # print(mat.keys())
        # print(info['name'], "shape: ", mat.shape, "inter: ", info['inter'])
# for i, info in enumerate(infos):
#     mat = sio.loadmat(info['path'])
#     if "02-043_OD_S2T.mat" not in info['name']:
#         print("跳过训练: ", info['name'], "shape: ", mat['S2T'].shape)
#         continue
#     print("开始训练: ", info['name'], "shape: ", mat['S2T'].shape)
#     data = mat['S2T']
#     train("HyperINR-temp-" + info['name'], device2, data)
    # if i % 2 == 0:
    #     device = device2
    #     name = f'{"HyperINR-" + info["name"]}_inr.pth'
    #     if name in res_before:
    #         print(f"跳过训练: {info['name']}, 已有结果")
    #         continue
    #     print(f"开始训练: {info['name']}, shape: {data.shape}")
    #     train("HyperINR-" + info['name'], device, data)
