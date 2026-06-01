# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

# from models import DiT_models
from my_mds.MyINR import HyperINRModel
from diffusion import create_diffusion
import wandb
from dataset.HyperDiffData import HyperDiffData
from matplotlib import pyplot as plt
from tqdm import tqdm

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    device = args.device
    torch.cuda.set_device(device)
    checkpoint_root = f"{args.save_dir}/{args.exp_name}"
    os.makedirs(checkpoint_root, exist_ok=True)
    if args.wandb:
        wandb.login(key="68e06a2142e71f48a9e7352a956bbcdb75675591")
        wandb.init(project="HyperINR", name=args.exp_name, config={})

    # Create model:
    latent_size = args.image_size // 8
    model = HyperINRModel(input_dim=2, hidden_dim=256, target_shapes=[(256, 256)] * 6, rank=128,
                          diffusion_mode=True).to(device)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    # 预测x0
    diffusion = create_diffusion(timestep_respacing="", predict_xstart=True)  # default: 1000 steps, linear noise schedule

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)

    # Setup data:
    dataset = HyperDiffData(args.data_path, inner_batch_size=2048)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    for epoch in range(args.epochs):
        print(f"Beginning epoch {epoch}...")
        for x, y, q, image_size in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            q = q.to(device)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            print("固定t")
            t = torch.ones_like(t) * 200  # 暂时固定

            model_kwargs = dict(y=y, q=q)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)

                avg_loss = avg_loss.item()

                model.eval()
                with torch.no_grad():
                    eval_img = model.eval_inr([(3, 4), (32, 33), (0, 128), (0, 128)], image_shape=image_size)
                    eval_img = (eval_img + 1) / 2 * 255
                    eval_img = eval_img.cpu().numpy().astype(np.uint8)
                    # plt.imshow(eval_img)
                    # plt.show()
                model.train()

                if args.wandb:
                    wandb.log({
                        "train_loss": avg_loss,
                        'inr_pred': [wandb.Image(eval_img, caption='inr_pred')],
                    })
                else:
                    plt.imshow(eval_img)
                    plt.show()

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_root}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")


    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    print("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/home/Data/zhangxiao/datasets/OLIVES/selected_mats/train-4D-resize(64, 128, 128)/')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--ckpt-every", type=int, default=200)
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default="/home/Data/zhangxiao/inr/models")
    parser.add_argument("--exp_name", type=str, default="pos-emb-fixed-t")
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()
    main(args)
