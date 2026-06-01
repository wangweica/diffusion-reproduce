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
采样一个INR示例
"""

# 定义INR模型
device = 'cuda:0'
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
path1 = "/home/Data/zhangxiao/datasets/OLIVES/selected_mats/test/"
path2 = "/home/Data/zhangxiao/datasets/OLIVES/selected_mats/train-4D-resize(64, 128, 128)/"
res_before_dir = '/home/Data/zhangxiao/inr/models/all_inr_bak/'
res_dir = '/home/Data/zhangxiao/inr/models/all_inr/'

target_save_root = "/home/Data/zhangxiao/inr/out/paper_show/"

res_before = os.listdir(res_before_dir) + os.listdir(res_dir)

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

infos = info1

for _, info in enumerate(infos):
    mat = sio.loadmat(info['path'])
    data = mat['S2T']
    if info["name"] != "02-043_OS_S2T.mat":
        continue
    model_name = f'{"HyperINR-" + info["name"]}_inr.pth'

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
    model.load_state_dict(torch.load(f"/home/Data/zhangxiao/inr/models/all_inr/{model_name}", map_location="cpu"))
    print(f"load model {model_path}")

    gif = []
    timeIdx = list(range(dataset.volume_sequence.shape[0]))
    labels = dataset.volume_sequence.unsqueeze(1).to(device)
    gif = []
    for i in tqdm(timeIdx, desc=f'test', unit='batch'):
        cube = []
        cube_gt = []
        save_file_name = f'{target_save_root}/{model_name}_inr_{i}.mat'

        # if os.path.exists(save_file_name):
        #     print(f"file {save_file_name} exist, skip")
        #     continue

        for s in range(29, 37):
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
            gt = postprocess_volume(labels, dataset.min_val, dataset.max_val).squeeze()
            slice = predictions.detach().squeeze()
            slice_gt = gt[i:i+1, s:s+1, :, :].detach().squeeze()

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
        cube = np.stack(cube, axis=0)
        cube_gt = np.stack(cube_gt, axis=0)
        sample = {
            "lr": cube,
            "hr": cube_gt,
        }
        sio.savemat(save_file_name, sample)


        gif.append(im_show)
        plt.imshow(im_show, cmap='gray')
        plt.show()
        plt.imshow(im_gt, cmap='gray')
        plt.show()


        # break
    # break
    # 保存gif
    # imageio.mimsave(f'/home/Data/zhangxiao/inr/out/im_temp/{model_name}_inr.gif', gif, loop=0)
    model.cpu()
    del model
    torch.cuda.empty_cache()

print("done")
