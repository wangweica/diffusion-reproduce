import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os

root = "/home/Data/zhangxiao/outs/"
methods = ["DDPM3D", "DDM", "VM", "cond-reg", "TLDM"]
# test_mat = "02-043_OS_S2T"
test_mat = "patient092"
def find_in_path(hint, path):
    files = os.listdir(path)
    for file in files:
        if file.endswith(".mat") and hint in file:
            return os.path.join(path, file)
    return None

for method in methods:
    path = os.path.join(root, method, "acdc")
    file_path = find_in_path(test_mat, path)
    if file_path is None:
        print(f"{test_mat} not found in {path}")
        continue
    retina_data = sio.loadmat(file_path)['data']
    if method == "DDM":
        print(retina_data.shape)
        retina_data = retina_data.transpose(0, 3, 1, 2)
    plt.imshow(retina_data[5, 0, :, :], cmap='gray')
    plt.show()
