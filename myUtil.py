'''
工具集
'''

import os
from natsort import natsorted
import scipy.io as sio
import numpy as np
import torch
import random
import scipy.ndimage as ndimage
def encode_acdc_filename(pid: str, week: str, pose: str, category :str='01', suffix :str='mat'):
    '''
    把这些属性编码为文件名
    '''
    return f'{category}.{pid}.{week}.{pose}.{suffix}'

def decode_acdc_filename(filename: str):
    '''
    把文件名(无后缀名)解码为属性
    '''
    category, pid, week, pose, suffix = filename.split('.')
    return category, pid, week, pose, suffix

def read_mats_tensor(paths):
    '''
    读取4D图像数据
    '''
    imgs_res = []
    label_res = []
    for path in paths:
        mat = sio.loadmat(path)
        label_res.append(mat['label'])
        imgs_res.append(mat['img'])

    imgs_res = np.stack(imgs_res, axis=0)
    label_res = np.stack(label_res, axis=0)
    # TODO: 修改这个缓存
    zoomed_path = os.path.join('saved_files', 'zoomed_imgs.npy')
    if os.path.exists(zoomed_path):
        imgs_res = np.load(zoomed_path)
    else:
        imgs_res = ndimage.zoom(imgs_res, (1, 1, 0.5714, 0.5714), order=3)
        np.save(zoomed_path, imgs_res)
    # label_res = ndimage.zoom(label_res, (1, 1, 0.5741, 0.5741), order=0)

    return torch.from_numpy(imgs_res).float(), torch.from_numpy(label_res).float()
def acdc_patients_group(path):
    '''
    按照病例分组, 输出4D图像的分组路径列表
    '''
    patients = os.listdir(path)
    patients = natsorted(patients)
    groups = {}
    for patient in patients:
        category, pid, week, pose, suffix = decode_acdc_filename(patient)
        key = f'{category}.{pid}.{pose}'
        if key not in groups:
            groups[key] = []
        groups[key].append(os.path.join(path, patient))

    # 按照星期排序
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: int(x.split('/')[-1].split('.')[-3][1:]))
    return groups


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    # torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    # torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def get_gray_histogram(img, height, width):
    """获取图像的灰度直方图"""
    gray = np.zeros(256)  # 保存各个灰度级（0-255）的出现次数

    for h in range(height):
        for w in range(width):
            gray[img[h][w]] += 1
    # 将直方图归一化, 即使用频率表示直方图
    gray /= (height * width)  # 保存灰度的出现频率，即直方图
    return gray

def get_gray_cumulative_prop(gray):
    """获取图像的累积分布直方图，即就P{X<=x}的概率
		- 大X表示随机变量
		- 小x表示取值边界
	"""
    cum_gray = []
    sum_prop = 0.
    for i in gray:
        sum_prop += i
        cum_gray.append(sum_prop)  # 累计概率求和
    return cum_gray

def run_histogram_match(img_src, img_dst):
    """运行图像直方图匹配, 把让dst的直方图适应src的直方图"""
    his1 = get_gray_histogram(img_src, img_src.shape[0], img_src.shape[1])  # 2.获取图像的灰度直方图
    his2 = get_gray_histogram(img_dst, img_dst.shape[0], img_dst.shape[1])
    cul_his1 = get_gray_cumulative_prop(his1)  # 3.获取图像的累积分布函数
    cul_his2 = get_gray_cumulative_prop(his2)

    # 寻找像素映射（累积概率，就进原则）
    new_index = []
    for each_gray in cul_his1:
        # 求出原直方图每一个灰度级累计概率在指定直方图上的灰度索引
        diff = list(abs(np.array(cul_his2 - each_gray)))
        closest_index = diff.index(min(diff))  # 索引代表对应填充的灰度级
        new_index.append(closest_index)

    # 填充像素
    height, width = img_src.shape
    new_img = np.zeros((height, width), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            new_img[h][w] = new_index[img_src[h][w]]

    # show_src = cv2.resize(img_src, (256, 256))
    # show_dst = cv2.resize(new_img, (256, 256))

    # plt.imshow(show_src)
    # plt.show()
    # plt.imshow(show_dst)
    # plt.show()
    return new_img

if __name__ == '__main__':
    groups = acdc_patients_group('/home/Data/zhangxiao/datasets/OLIVES/01-mats-post/train')
    print(groups)