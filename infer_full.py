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
from train_ldm import LDM, get_parser
from my_mds.utils_pack.util_p import load_network_3
from core import *
from einops import rearrange
from torchmetrics.image import (PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)

def calculate_nmse(processed_image, original_image):
    sum_diff = torch.sum((original_image - processed_image) ** 2)
    nmse = sum_diff / torch.sum(original_image ** 2)
    # print(torch.sum(original_image ** 2))
    return nmse
 # input (N,C,H,W)
SSIM = StructuralSimilarityIndexMeasure(data_range=(0, 255))
# input (N,C,H,W)
PSNR = PeakSignalNoiseRatio(data_range=(0, 255))

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize((0.5,), (0.5,))
])
# 定义INR模型
device = 'cuda:0'
parser = get_parser()
opts = parser.parse_args()
ldm = LDM(opts).to(device)
ckpt_path = "/home/Data/zhangxiao/inr/out/inr_refine_log/vaeldm_2d_batch_cond_fulltuning_num_head_4/lightning_logs/version_15/checkpoints/last.ckpt"
load_network_3(ldm, ckpt_path, device)


path1 = "/home/Data/zhangxiao/datasets/OLIVES/selected_mats/test/"
path2 = "/home/Data/zhangxiao/datasets/OLIVES/selected_mats/train-4D-resize(64, 128, 128)/"
res_before_dir = '/home/Data/zhangxiao/inr/models/all_inr_bak/'
res_dir = '/home/Data/zhangxiao/inr/models/all_inr/'

gif_out_path = "/home/Data/zhangxiao/inr/out/im_temp/"

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

infos = info1 + info2
data = None
model_name = None
s = 56
for _, info in enumerate(infos):
    if info['name'] != "02-015_OD_S2T.mat":
        continue

    print(f"processing {info['name']}")
    mat = sio.loadmat(info['path'])
    data = mat['S2T']
    model_name = f'{"HyperINR-" + info["name"]}_inr.pth'

dataset = TIF_Dataset(None, device, inner_batch_size=1, img=data, util=True)

model = HyperINRModel(input_dim=2, hidden_dim=256, target_shapes=[(256, 256)] * 6, rank=64).to(device)
model.eval()

model_path = os.path.join(res_before_dir, model_name)
T = 10
if not os.path.exists(model_path):
    model_path = os.path.join(res_dir, model_name)
    T = data.shape[0]

# 模型评估
model.load_state_dict(torch.load(model_path, map_location="cpu"))
print(f"load model {model_path}")

gif = []
timeIdx = np.linspace(0, T-1, T, dtype=np.int64)
labels = dataset.volume_sequence.unsqueeze(1).to(device)

for i in tqdm(timeIdx, desc=f'test', unit='batch'):
    # 前后两帧合并输入
    cond = torch.cat([labels[0:1], labels[T - 1:T]], dim=1)
    coords = dataset.inr_input_util[(i, i+1, 301), s:s+8, :, :]
    coords = dataset.inr_input_util.normalize(coords)
    coords = torch.from_numpy(coords).float()
    # print("coords:", coords.shape)

    coords, cond = dataset.toDevice([coords, cond], device)
    with torch.no_grad():
        model.eval()
        outputs = model(y=cond, q=coords)
        outputs = outputs.reshape(8, 128, 128)
        # print("eval output:", outputs.mean(), outputs.std())

    # (1, 1, 64, 224, 224)
    predictions = postprocess_volume(outputs, dataset.min_val, dataset.max_val)
    slice = predictions.detach().squeeze()
    gt = postprocess_volume(labels, dataset.min_val, dataset.max_val).squeeze()
    slice_gt = gt[i:i + 1, s:s+8, :, :].detach().squeeze()

    im_show = slice.cpu().numpy().astype(np.uint8)
    im_gt = slice_gt.cpu().numpy().astype(np.uint8)
    # plt.imshow(im_show[3], cmap='gray')
    # plt.show()
    plt.imshow(im_gt[3], cmap='gray')
    plt.show()
    im_z = []
    for s_8 in range(8):
        im_show_8 = im_show[s_8]
        normed_im = transform(image=im_show_8)['image']
        x = transforms.ToTensor()(normed_im).to(device)
        z = ldm.encode_first_stage(x)
        im_z.append(z.detach().squeeze())

    im_z = torch.stack(im_z, dim=0).unsqueeze(0)
    ldm_input = rearrange(im_z, '1 b c h w -> b c h w')
    x_samples = ldm.sample(c=ldm_input, batch_size=ldm_input.shape[0], return_intermediates=False, clip_denoised=False)
    img_samples = ldm.decode_first_stage(x_samples).to('cpu').squeeze().numpy()
    img_samples = (img_samples + 1) * 127.5
    im_res = img_samples[3]
    im_tensor = torch.from_numpy(im_res[None, None])
    im_gt = transform(image=im_gt[3])['image']
    im_gt_tensor = torch.from_numpy(im_gt[None, None])
    im_gt_tensor = (im_gt_tensor + 1) * 127.5

    psnr = PSNR(im_tensor, im_gt_tensor)
    ssim = SSIM(im_tensor, im_gt_tensor)
    nmse = calculate_nmse(im_tensor, im_gt_tensor)

    print(f"psnr: {psnr}, ssim: {ssim}, nmse: {nmse}")
    plt.imshow(im_res, cmap='gray')
    plt.show()
#     gif.append(im_show)
#
# imageio.mimsave(os.path.join(gif_out_path, f"{model_name.split('.')[0]}.gif"), gif, loop=0)