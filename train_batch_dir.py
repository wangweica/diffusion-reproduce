import os
import scipy.io as sio
import numpy as np
from tif_train_func import train
from natsort import natsorted
from myUtil import *
data_path = "/home/Data/zhangxiao/datasets/lung2-mats/train"
groups = {}
for file in os.listdir(data_path):
    if file.endswith(".mat"):
        name = file.replace(".mat", "")
        pname, inter = name.split("_")
        inter = int(inter)
        if pname not in groups:
            groups[pname] = []
        groups[pname].append({'name': name, 'path': os.path.join(data_path, file), 'inter': inter})


device2 = torch.device("cuda:1")
idx = 1
for pname, info_list in groups.items():
    if idx % 2 != 0:
        device2 = torch.device("cuda:1")
        idx += 1
        continue
    else:
        device2 = torch.device("cuda:2")
        idx += 1


    info_list = natsorted(info_list, key=lambda x: x['inter'])
    fourD_data = []
    # "/home/Data/zhangxiao/datasets/lung2-mats/train/case2_3.mat"
    for i, info in enumerate(info_list):
        mat = sio.loadmat(sio.loadmat(info['path'])['inter_image'][0])['data']
        fourD_data.append(mat)
    fourD_data = np.stack(fourD_data, axis=0)
    # 归一化
    fourD_data = (fourD_data - fourD_data.min()) / (fourD_data.max() - fourD_data.min())
    fourD_data *= 255
    print(pname, "shape: ", fourD_data.shape, "device: ", device2, "min_max:", fourD_data.min(), fourD_data.max())
    train(pname, device2, fourD_data, model_save_path="/home/Data/zhangxiao/inr/models/all_inr_dir/")
