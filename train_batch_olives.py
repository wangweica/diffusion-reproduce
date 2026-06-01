import os
import scipy.io as sio
import numpy as np
from tif_train_func import train
from natsort import natsorted

path1 = "/home/Data/zhangxiao/datasets/OLIVES/selected_mats/test/"
path2 = "/home/Data/zhangxiao/datasets/OLIVES/selected_mats/train-4D-resize(64, 128, 128)/"
res_before_dir = '/home/Data/zhangxiao/inr/models/all_inr_bak/'
res_dir = '/home/Data/zhangxiao/inr/models/all_inr/'
res_before = os.listdir(res_before_dir) + os.listdir(res_dir)

device1 = "cuda:0"
device2 = "cuda:0"
files1 = natsorted(os.listdir(path1))
files2 = natsorted(os.listdir(path2))

info1 = []
for f in files1:
    info1.append({
        "path": path1 + f,
        "name": f
    })

info2 = []
for f in files2:
    info2.append({
        "path": path2 + f,
        "name": f
    })

infos = info1 + info2
# 加锁
p1 = False
p2 = False
for i, info in enumerate(infos):
    mat = sio.loadmat(info['path'])
    if "02-043_OD_S2T.mat" not in info['name']:
        print("跳过训练: ", info['name'], "shape: ", mat['S2T'].shape)
        continue
    print("开始训练: ", info['name'], "shape: ", mat['S2T'].shape)
    data = mat['S2T']
    train("HyperINR-temp-" + info['name'], device2, data)
    # if i % 2 == 0:
    #     device = device2
    #     name = f'{"HyperINR-" + info["name"]}_inr.pth'
    #     if name in res_before:
    #         print(f"跳过训练: {info['name']}, 已有结果")
    #         continue
    #     print(f"开始训练: {info['name']}, shape: {data.shape}")
    #     train("HyperINR-" + info['name'], device, data)
