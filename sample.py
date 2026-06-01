# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
import argparse
from my_mds.MyINR import HyperINRModel
from dataset.HyperDiffData import HyperDiffData
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = args.device

    # Load model:
    model = HyperINRModel(input_dim=2, hidden_dim=256, target_shapes=[(256, 256)] * 6, rank=32,
                          diffusion_mode=True).to(device)

    # 加载模型参数
    state_dict = torch.load(args.ckpt, map_location="cpu")["model"]
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    dataset = HyperDiffData(args.data_path, inner_batch_size=2048)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    x, y, q, image_size = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    q = q.to(device)
    z = torch.randn_like(x)
    model_kwargs = dict(y=y, q=q)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )

    # 使用最后一步的INR作为结果
    with torch.no_grad():
        eval_img = model.eval_inr([(3, 4), (32, 33), (0, 128), (0, 128)], image_shape=image_size)
        eval_img = (eval_img + 1) / 2 * 255
        eval_img = eval_img.cpu().numpy().astype(np.uint8)
        plt.imshow(eval_img)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--ckpt", type=str,
                        default="/home/Data/zhangxiao/inr/models/HyperINR_diffusion/0047600.pt")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--data-path", type=str,
                        default='/home/Data/zhangxiao/datasets/OLIVES/selected_mats/train-4D-resize(64, 128, 128)/')

    args = parser.parse_args()
    main(args)