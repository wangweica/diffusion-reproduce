import numpy as np
import torch
import torch.nn as nn
from myUtil import *
from torch.utils.data import DataLoader, Dataset
class INRInputData:
    def __init__(self, data, coord_shape, B_gauss=None, maxVal=0, minVal=0):
        """
        初始化INRInputData类
        用于把4D数据转换为坐标数据，方便后续的训练和预测
        参数:
        data: 4D数据，形状为 (t, x, y, z)
        """
        self.data = data
        self.shape = coord_shape
        self.maxVal = maxVal
        self.minVal = minVal
        self.B_gauss = B_gauss
        self.vae = None

    def decompose_tuple(self, tup):
        # 解压切片中的元组为序列, 一般用于构造非整数的下标(也就是timestep)
        # start, end, step = tup
        return np.arange(*tup).tolist()

    def __getitem__(self, key):
        """
        获取指定索引或切片范围的坐标
        参数:
        key: 索引或切片范围
        返回:
        (Batch, 4) 的坐标数组
        """
        if isinstance(key, tuple):
            # 确保传入的key是一个四维索引或切片
            if len(key) != 4:
                raise IndexError("需要四个索引或切片.")
            # 处理每个维度的切片
            slices = [range(*k.indices(self.shape[i])) if isinstance(k, slice)
                      else self.decompose_tuple(k)
                      for i, k in enumerate(key)]

        else:
            raise TypeError("索引必须是一个包含四个元素的元组.")

        # 创建网格坐标
        meshgrid = np.meshgrid(*slices, indexing='ij')

        # 将网格坐标转换为(Batch, 4)的形状
        coords = np.stack(meshgrid, axis=-1).reshape(-1, 4)

        return coords

    def normalize(self, array):
        """
        数据归一化到指定范围
        :param array: (Batch, 4) 或者 (4,) 的数据
        :return:
        """
        if isinstance(array, torch.Tensor):
            out = array.clone().detach()
        else:
            out = array.copy().astype(np.float64)

        # 首先将数据归一化到[0, 1]的范围
        for i, max_val in enumerate(self.shape):
            denom = max_val - 1 if max_val > 1 else 1.0
            if len(array.shape) == 2:
                out[:, i] = array[:, i] / denom
            elif len(array.shape) == 1:
                out[i] = array[i] / denom
            else:
                raise ValueError("输入数据必须是(Batch, 4)或(4,)的形状.")

        return out

    def __repr__(self):
        return f"INRInputData(shape={self.shape})"

class VolumeDataset(Dataset):
    def __init__(self, volume_sequence, B_gauss, inr_data=None):
        self.volume_sequence = preprocess_volume(volume_sequence)
        # z_min_coords = encode_layer_volume(volume_sequence)
        # try_ = decode_layer_coord(z_min_coords, volume_sequence.shape)

        self.B_gauss = B_gauss
        self.time_steps, self.depth, self.height, self.width = volume_sequence.shape
        # 转换为坐标数据
        self.inr_data = inr_data
        # self.inr_data = INRInputData(volume_sequence)
        # 训练全部的数据
        coords = self.inr_data[:, :, :, :]
        # 转换为torch tensor
        self.data = torch.tensor(coords, dtype=torch.float32)
        print(f'数据集初始化完成, 数据量: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixel_idx = self.data[idx]
        t, x, y, z = pixel_idx.clone().detach().numpy()

        # 输入归一化[0,1]后的坐标
        pixel_idx = self.inr_data.normalize(pixel_idx)

        # 映射成傅里叶特征
        input_index = input_mapping(pixel_idx, self.B_gauss)
        # input_index = pixel_idx
        # 输出对应位置的体素值
        label = self.volume_sequence[int(t), int(x), int(y), int(z)].unsqueeze(0)

        return input_index, label

class VolumeLatentDataset(Dataset):
    def __init__(self, volume_sequence, B_gauss, device):
        self.volume_sequence = preprocess_volume(volume_sequence)
        # z_min_coords = encode_layer_volume(volume_sequence)
        # try_ = decode_layer_coord(z_min_coords, volume_sequence.shape)

        self.B_gauss = B_gauss
        self.time_steps, self.depth, self.feat_ch, self.height, self.width = volume_sequence.shape
        # 由于latent特征都是离散的, 因此不可以把latent的坐标输入
        # 我们只能假设时间维度是连续的 (尽管数据不连续)
        coords = np.linspace(0, 1, self.time_steps)
        coords = torch.from_numpy(coords).float().to(device)
        print(f'数据集初始化完成, 数据量: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixel_idx = self.data[idx]
        t, x, y, z = pixel_idx.clone().detach().numpy()

        # 输入归一化[0,1]后的坐标
        pixel_idx = self.inr_data.normalize(pixel_idx)

        # 映射成傅里叶特征
        input_index = input_mapping(pixel_idx, self.B_gauss)
        # input_index = pixel_idx
        # 输出对应位置的特征
        label = self.volume_sequence[int(t), int(x), :, int(y), int(z)]

        return input_index, label

def preprocess_volume(tensor):
    # 找出张量的最大值和最小值
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)

    # 将张量的值归一化到[0, 1]的范围
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    return normalized_tensor


def postprocess_volume(tensor, min_val=0, max_val=6):
    # 将张量的值反归一化到原始值范围
    original_tensor = (tensor * (max_val - min_val)) + min_val
    return original_tensor

def input_mapping(x, B_gauss, scale=1.0):
    '''
    把坐标映射到傅里叶特征
    '''
    # Three different scales of Gaussian Fourier feature mappings
    B = B_gauss * scale
    x_proj = (2. * np.pi * x) @ B.T
    return torch.concatenate([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

if __name__ == '__main__':
    # 示例使用
    data = np.random.rand(10, 20, 30, 40)  # 生成一个4D数组
    inr_input = INRInputData(data)

    # 获取特定索引或切片的坐标
    # coords = inr_input[1:3, 5:10, 0:15, 10:20]
    coords = inr_input[:, :, :, :]
    print(coords)
    print(coords.shape)  # 输出 (time_steps * depth * height * width, 4)
    # pass