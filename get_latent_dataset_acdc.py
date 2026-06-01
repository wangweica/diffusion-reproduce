import torch
import os
import scipy.io as sio
import numpy as np
from core import *
data_path = "/home/Data/zhangxiao/inr/out/inr2img_dataset_acdc/"
target_path = "/home/Data/zhangxiao/inr/out/inr2img_latent_dataset_acdc/"
os.makedirs(target_path, exist_ok=True)
os.makedirs(os.path.join(target_path, 'inr'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'gt'), exist_ok=True)
model = VAE()
model.eval()
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model.to(device)

def infer_one_cube(img_sequence, name):
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize((0.5,), (0.5,))
    ])

    S, H, W = img_sequence.shape

    cube_res = []
    for s in range(S):
        x = img_sequence[s].numpy()
        x = transform(image=x)['image']
        x = transforms.ToTensor()(x)

        x = x.repeat(1, 3, 1, 1)
        x = x.to(device)

        with torch.no_grad():
            z = model.encode(x)
            print(z.shape)
            cube_res.append(z.squeeze().cpu())

            # np.save(os.path.join(cube_dir, f'{cube_name}_{i}.npy'), z.cpu().numpy())

            # xrec = model.decode(z)
            # xrec = torch.mean(xrec, dim=1, keepdim=True)
            #
            # print(x.max(), x.min())
            # plt.imshow(x.cpu().numpy()[0, 0])
            # plt.show()
            #
            # plt.imshow(xrec.cpu().numpy()[0, 0])
            # plt.show()
            # pass
    cube_res = torch.stack(cube_res)

    np.save(os.path.join(target_path, f'{name}.npy'), cube_res.cpu().numpy())

for f in os.listdir(data_path):
    data = sio.loadmat(os.path.join(data_path, f))
    lr = data['lr']
    hr = data['hr']
    lr = torch.from_numpy(lr).float()
    hr = torch.from_numpy(hr).float()
    print(lr.shape, hr.shape)
    infer_one_cube(lr, 'inr/' + f)
    infer_one_cube(hr, 'gt/' + f)
    print(f"Finish {f}")

