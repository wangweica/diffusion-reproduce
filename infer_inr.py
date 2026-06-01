import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from myUtil import *
from tqdm import tqdm
import wandb
from dataset.INR_data import VolumeDataset, postprocess_volume, input_mapping, INRInputData, VolumeLatentDataset, preprocess_volume
from core import adjust_learning_rate, evaluate
import matplotlib.pyplot as plt
import imageio
from my_mds.MyINR import INRModel_v2 as INRModel
from my_mds.INRs import HyperINRModel
from dataset.TIF_Dataset import TIF_Dataset

"""
用训练好的INR构造INR图像到真实图像的数据集.
"""

# 定义INR模型
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f'Using CUDA device: {torch.cuda.current_device()}')
else:
    device = torch.device('cpu')
    print('CUDA unavailable, using CPU')
data_path = 'data'
model_save_path = 'saved_models'

is_train = True
is_wandb = False if is_train else False

RANDOM_SEED = 42  # any random number
num_epochs = 1000
base_lr = 0.00003
save_interval = 100
inner_batch_size = 1
set_seed(RANDOM_SEED)
path1 = "./data/"
path2 = "./data/"
possible_roots = [
    os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..')),
    os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', '..')),
    os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', '..', '..')),
]
base_dir = None
for root in possible_roots:
    if os.path.isdir(os.path.join(root, 'models', 'all_inr')):
        base_dir = root
        break
if base_dir is None:
    base_dir = possible_roots[-1]
    print('Warning: could not find models/all_inr under expected parent directories; using', base_dir)

res_before_dir = os.path.join(base_dir, 'models', 'all_inr_bak')
res_dir = os.path.join(base_dir, 'models', 'all_inr')

dataset_save_root = os.path.join(os.getcwd(), 'inr2img_dataset')
os.makedirs(dataset_save_root, exist_ok=True)

res_before = []
if os.path.isdir(res_before_dir):
    res_before = os.listdir(res_before_dir)
if os.path.isdir(res_dir):
    res_before += os.listdir(res_dir)

files1 = []
files2 = []
if os.path.isdir(path1):
    files1 = natsorted(os.listdir(path1))
if os.path.isdir(path2):
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


def get_mat_volume(mat):
    for key in ['S2T', 'img', 'data', 'label']:
        if key in mat:
            return mat[key]
    for key, value in mat.items():
        if not key.startswith('__'):
            return value
    raise KeyError('MAT file does not contain a usable volume key')

for _, info in enumerate(infos):
    mat = sio.loadmat(info['path'])
    data = get_mat_volume(mat)
    if data.ndim == 3:
        data = data[np.newaxis, ...]
    file_base = os.path.splitext(info['name'])[0]
    model_name = f'{"HyperINR-" + file_base}_inr.pth'

    # if model_name not in res_before:
    #     print(f"model {model_name} not exist, skip")
    #     continue

    model_path = os.path.join(res_before_dir, model_name)
    T = 10
    if not os.path.exists(model_path):
        model_path = os.path.join(res_dir, model_name)
        T = data.shape[0]

    dataset = TIF_Dataset(None, device, inner_batch_size=inner_batch_size, img=data, util=True)

    model = HyperINRModel(input_dim=2, hidden_dim=256, target_shapes=[(256, 256)] * 6, rank=64).to(device)
    model.eval()

    # 模型评估
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    print(f"load model {model_path}")

    gif = []
    timeIdx = list(range(dataset.volume_sequence.shape[0]))
    labels = dataset.volume_sequence.unsqueeze(1).to(device)
    gif = []
    for i in tqdm(timeIdx, desc=f'test', unit='batch'):
        cube = []
        cube_gt = []
        save_file_name = f'{dataset_save_root}/{model_name}_inr_{i}.mat'

        # if os.path.exists(save_file_name):
        #     print(f"file {save_file_name} exist, skip")
        #     continue

        for s in range(64):
            # 前后两帧合并输入
            cond = torch.cat([labels[0:1], labels[T-1:T]], dim=1)
            # coords = dataset.inr_input_util[(timesteps[i], timesteps[i]+1, 301), (42, 43, 43), :, :]
            # coords = dataset.inr_input_util.normalize(coords)
            # coords = torch.from_numpy(coords).float()
            coords = dataset.get_slice_coords([(i, i+1), (s, s+1), (0, 128), (0, 128)])
            coords, cond = dataset.toDevice([coords, cond], device)
            with torch.no_grad():
                model.eval()
                outputs = model(y=cond, q=coords)
                outputs = outputs.reshape(128, 128)
                # print("eval output:", outputs.mean(), outputs.std())

            # (1, 1, 64, 224, 224)
            predictions = postprocess_volume(outputs, dataset.min_val, dataset.max_val)
            gt = postprocess_volume(labels, dataset.min_val, dataset.max_val)
            if gt.ndim == 5 and gt.shape[1] == 1:
                gt = gt.squeeze(1)
            slice = predictions.detach().squeeze()
            slice_gt = gt[i:i+1, s:s+1, 0:128, 0:128].detach().squeeze()

            im_show = slice.cpu().numpy().astype(np.uint8)
            im_gt = slice_gt.cpu().numpy().astype(np.uint8)
            # sample = {
            #     "lr": im_show,
            #     "hr": im_gt,
            # }
            # sio.savemat(f'{dataset_save_root}/{model_name}_inr_{i}_{s}.mat', sample)
            cube.append(im_show)
            cube_gt.append(im_gt)


        # 保存cube
        # cube = np.stack(cube, axis=0)
        # cube_gt = np.stack(cube_gt, axis=0)
        # sample = {
        #     "lr": cube,
        #     "hr": cube_gt,
        # }
        # sio.savemat(save_file_name, sample)


        gif.append(im_show)
        if i == 0 and s == 0:
            print('Sample output shapes:', im_show.shape, im_gt.shape)
            imageio.imsave(os.path.join(dataset_save_root, f'{model_name}_sample_pred.png'), im_show)
            imageio.imsave(os.path.join(dataset_save_root, f'{model_name}_sample_gt.png'), im_gt)


        # break
    # break
    # 保存gif
    # imageio.mimsave(f'/home/Data/zhangxiao/inr/out/im_temp/{model_name}_inr.gif', gif, loop=0)
    model.cpu()
    del model
    torch.cuda.empty_cache()

print("done")
