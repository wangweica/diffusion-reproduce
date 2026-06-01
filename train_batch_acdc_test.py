import os
import scipy.io as sio
import numpy as np
from tif_train_func import train
from natsort import natsorted
from myUtil import *
data_path = "/home/Data/zhangxiao/datasets/acdc_dataset/mats-inter/test"
groups = {}

device2 = torch.device("cuda:1")
idx = 1
for fname in os.listdir(data_path):
    if idx % 2 != 0:
        device2 = torch.device("cuda:1")
        idx += 1
        continue
    else:
        device2 = torch.device("cuda:0")
        idx += 1


    fourD_data = sio.loadmat(os.path.join(data_path, fname))['4D_lable']
    fourD_data = np.transpose(fourD_data, (0, 3, 1, 2))
    # print(fourD_data.keys())
    pname = fname.split('.')[0]
    # 归一化
    fourD_data = (fourD_data - fourD_data.min()) / (fourD_data.max() - fourD_data.min())
    fourD_data *= 255
    print(pname, "shape: ", fourD_data.shape)
    train(pname, device2, fourD_data, model_save_path="/home/Data/zhangxiao/inr/models/all_inr_acdc_test/")

