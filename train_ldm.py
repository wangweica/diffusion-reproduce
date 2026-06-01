# -*- coding:utf-8 -*-
import os
import shutil
import os
import argparse, os, sys, glob
curPath = os.path.abspath(os.path.dirname(__file__))
# sys.path.append("/home/xiebaoye/latent-diffusion/cond_ldm/")

import json
# -*- coding:utf-8 -*-
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_only
from diffusers import AutoencoderKL
from my_mds.ldm.ldm_mds.ema_xby import LitEma
from my_mds.networks.openaimodel_cond_cross_batch_ldm_v1 import UNetModel, AttentionBlock_adaptor, SpatialTransformer
from my_mds.utils_pack.util_for_opencv_diffusion import DDPM_base, disabled_train, LambdaLinearScheduler
# from einops import rearrange
# from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from my_mds.utils_pack.util_p import load_network_3
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from natsort import natsorted
from einops import rearrange
from core import *
"""
full tuning

using batch cross-attention
"""

GLOBAL_SCALE = 1.0 / 12

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--command", default="fit")
    parser.add_argument("--exp_name", default='OLIVES_vaeldm_2d_batch_cond_fulltuning_num_head_4')
    parser.add_argument('--result_root', type=str, default=os.path.join(os.getcwd(), 'train_logs'))
    parser.add_argument('--inr_data_path', type=str, default=os.path.join(os.getcwd(), 'inr2img_latent_dataset'))

    parser.add_argument('--first_stage_ckpt', type=str,
                        default='')

    # train args
    parser.add_argument("--latent_size", nargs=2, type=int, default=[64, 64],
                        help="latent spatial size, e.g. --latent_size 64 64")
    parser.add_argument("--latent_channel", default=4)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("--pin_memory", default=False)
    parser.add_argument("--base_lr", type=float, default=3.5e-5)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    # lightning args
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--eval_save_every_n_epoch", type=int, default=20)
    parser.add_argument("--limit_train_batches", type=int, default=10000)
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='auto', choices=['auto', 'cpu', 'gpu', 'cuda'])
    parser.add_argument('--precision', default=32)
    parser.add_argument('--devices', default=None, type=int,
                        help='Number of devices to use for training. Leave empty for auto selection.')
    parser.add_argument('--reproduce', type=int, default=False)
    return parser


def get_pathes(oct_root):
    cube_names = natsorted(os.listdir(oct_root))
    pathes = []
    for cube_name in cube_names:
        pathes.append(os.path.join(oct_root, cube_name))
    return pathes


def get_multi_pathes(oct_roots):
    pathes = []
    for oct_root in oct_roots:
        pathes += get_pathes(oct_root)
    return pathes


def main(opts):
    # data & tio args
    inr_data_path = opts.inr_data_path
    print('train_ldm using inr_data_path:', inr_data_path)
    print('cwd:', os.getcwd())
    # cond_root1 = '/home/Data/xiebaoye/model_data/datasets/fusing/100edema/cond'
    # cond_root2 = '/home/Data/xiebaoye/model_data/datasets/fusing/100edema/cond2_code'
    # latent_2D_root = '/home/Data/xiebaoye/100edema/latent'
    # train_name_json = '/home/Data/xiebaoye/model_data/datasets/fusing/100edema/train.jsonl'
    # interval_len = 5

    # cond_root1 = '/home/Data/xiebaoye/model_data/datasets/fusing/2017challenge/cond'
    # cond_root2 = '/home/Data/xiebaoye/model_data/datasets/fusing/2017challenge/cond2_code'
    # latent_2D_root = '/home/Data/xiebaoye/2017challenge/Cirrus/latent'
    # train_name_json = '/home/Data/xiebaoye/model_data/datasets/fusing/2017challenge/train.jsonl'
    # interval_len = 5


    # with open(train_name_json, "r") as f:
    #     train_cube_names = json.load(f)
    # if "100edema" in train_name_json:
    #     with open(train_name_json.replace("100edema","2017challenge"), "r") as f:
    #         train_cube_names.extend(json.load(f))

    inr_data_path = opts.inr_data_path
    if not os.path.isdir(inr_data_path):
        raise FileNotFoundError(
            f"INR latent dataset path not found: {inr_data_path}\n"
            f"Please create the folder and put your latent .npy files under '{os.path.join(inr_data_path, 'inr')}' and '{os.path.join(inr_data_path, 'gt')}'.")

    train_set = unlabeled_Dataset_inr(inr_data_path)
    print('train_set: ', len(train_set))

    # GPU/CPU accelerator selection
    if opts.accelerator in ('auto', 'cuda'):
        if torch.cuda.is_available():
            opts.accelerator = 'gpu'
        else:
            opts.accelerator = 'cpu'
    if opts.accelerator == 'gpu' and not torch.cuda.is_available():
        print('WARNING: --accelerator gpu requested but no CUDA GPU detected. Falling back to CPU.')
        opts.accelerator = 'cpu'

    if opts.devices is None:
        if opts.accelerator == 'gpu' and torch.cuda.is_available():
            opts.devices = min(1, torch.cuda.device_count())
        else:
            opts.devices = 1

    print(f"Using accelerator={opts.accelerator}, devices={opts.devices}")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")

    if isinstance(opts.latent_size, (list, tuple)):
        opts.latent_size = tuple(opts.latent_size)
    else:
        try:
            opts.latent_size = tuple(int(x) for x in str(opts.latent_size).split())
        except Exception:
            opts.latent_size = (64, 64)

    # infer latent spatial size from first sample to keep sampling consistent
    first_sample = train_set[0]
    sample_latent_size = tuple(first_sample['gt_data'].shape[2:])
    if opts.latent_size != sample_latent_size:
        print(f"Adjusting latent_size from {opts.latent_size} to dataset spatial size {sample_latent_size}")
        opts.latent_size = sample_latent_size

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=opts.num_workers,
                              pin_memory=opts.pin_memory)
    # for batch in train_loader:
    #     print(batch['latent_2D'].shape)
    #     break

    model = LDM(opts)
    ckpt_callback = ModelCheckpoint(save_last=True, filename="model-{epoch}", save_top_k=-1,
                                    every_n_epochs=opts.eval_save_every_n_epoch)
    trainer = pl.Trainer(max_epochs=opts.max_epochs, limit_train_batches=opts.limit_train_batches,
                         accelerator=opts.accelerator, precision=opts.precision, devices=opts.devices,
                         deterministic=opts.deterministic,  # strategy="ddp",
                         default_root_dir=opts.default_root_dir, profiler=opts.profiler, benchmark=opts.benchmark,
                         callbacks=[ckpt_callback, TQDMProgressBar(refresh_rate=10)])

    # ckpt_path = '/home/Data/xiebaoye/cldm_train_logs/ldm_train/logs/vqldm_2d_uncond_multiSlice_crossBatch_v2/lightning_logs/version_2/checkpoints/model-epoch=499.ckpt'
    # load_network_3(model, ckpt_path, model.device)
    # ckpt_path = "/home/Data/xiebaoye/cldm_train_logs/ldm_train/logs/vqldm_2d_uncond_multiSlice_crossBatch_v2_num_head_4/lightning_logs/version_2/checkpoints/last.ckpt"

    # ckpt_path = ""
    # load_network_3(model, ckpt_path, model.device)

    trainer.fit(model=model, train_dataloaders=train_loader)


class unlabeled_Dataset2(Dataset):

    def __init__(self, cond_root1, cond_root2, latent_2D_root, interval_len, cube_names):

        self.cond_root1 = cond_root1
        self.cond_root2 = cond_root2
        self.latent_2D_root = latent_2D_root
        self.cube_names = cube_names
        self.interval_len = interval_len

    def read_label(self, path):
        label = np.load(path)
        label = torch.from_numpy(label).float()
        # label /= 255
        label = label.unsqueeze(0)
        return label

    def __getitem__(self, index):
        name = self.cube_names[index]
        cond1_path = self.cond_root1
        cond2_path = self.cond_root2
        latent_2D_path = self.latent_2D_root

        if "cube_z" not in name:
            cond1_path = cond1_path.replace("100edema", "2017challenge")
            latent_2D_path = "/home/Data/xiebaoye/2017FluidChallenge/Cirrus/latent"
            cond2_path = cond2_path.replace("100edema", "2017challenge")

        # latent_2D = np.load(os.path.join(self.latent_2D_root, name + '.npy'))
        # latent_2D = torch.from_numpy(latent_2D).float()

        # cond1 = self.read_label(os.path.join(self.cond_root1, name + '.npy'))
        cond2 = np.load(os.path.join(cond2_path, name + '.npy'))
        cond2 = torch.from_numpy(cond2)[0]

        l = 128
        start_index = torch.randint(0, l - self.interval_len + 1, (1,)).item()
        latent_selected = np.zeros((4, self.interval_len, 120, 64))
        # latent_selected = latent_2D[:, start_index:start_index + self.interval_len,:,:]

        cond1_selected = np.zeros((1, self.interval_len, 480, 256))
        for i in range(start_index, start_index + self.interval_len):
            latent_frame = np.load(os.path.join(latent_2D_path, name, str(i + 1) + '.npy'))
            latent_frame = torch.from_numpy(latent_frame).float()
            latent_selected[:, i - start_index, :, :] = latent_frame

            cond1_frame = self.read_label(os.path.join(cond1_path, name, str(i + 1) + '.npy'))
            cond1_selected[0, i - start_index, :, :] = cond1_frame

        # cond1_selected = cond1[:, start_index:start_index + self.interval_len,:,:]
        cond2_selected = cond2[start_index:start_index + self.interval_len]

        return {'cond1': cond1_selected, 'cond2': cond2_selected,
                'latent_2D': latent_selected, 'name': name}

    def __len__(self):
        return len(self.cube_names)

class unlabeled_Dataset_inr(Dataset):
    def __init__(self, data_root):
        self.inr_data_root = os.path.join(data_root, 'inr')
        self.gt_data_root = os.path.join(data_root, 'gt')
        self.cube_names = natsorted(os.listdir(self.inr_data_root))

    def __getitem__(self, index):
        name = self.cube_names[index]
        # [F, C, H, W]
        inr_data = np.load(os.path.join(self.inr_data_root, name))
        inr_data = torch.from_numpy(inr_data).float()

        gt_data = np.load(os.path.join(self.gt_data_root, name))
        gt_data = torch.from_numpy(gt_data).float()
        # TODO: 偷懒写法，后续需要修改
        inr_data *= GLOBAL_SCALE
        gt_data *= GLOBAL_SCALE

        # print(gt_data.min(), gt_data.max())
        # print(inr_data.min(), inr_data.max())
        # print('-----------')
        frames = 8
        # 选取8帧
        start_index = torch.randint(0, inr_data.shape[0] - frames + 1, (1,)).item()
        # print(start_index)
        inr_data_selected = inr_data[start_index:start_index + frames, :, :, :]
        # gt对应的8帧
        gt_data_selected = gt_data[start_index:start_index + frames, :, :, :]
        # gt的第一和最后一帧
        se_frame = torch.cat([gt_data[0:1, :, :, :], gt_data[-1:, :, :, :]], dim=1)
        se_frame = se_frame.repeat(frames, 1, 1, 1)
        return {'inr_data': inr_data_selected, 'gt_data': gt_data_selected, 'se_frame': se_frame, 'name': name}

    def __len__(self):
        return len(self.cube_names)
class LDM(DDPM_base):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.save_hyperparameters()

        unet_config = {'image_size': opts.latent_size, 'in_channels': opts.latent_channel * 4,
                       'out_channels': opts.latent_channel, 'model_channels': 192,
                       'attention_resolutions': [2, 4, 8], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4],
                       "num_classes": None,
                       'num_heads': 4, 'use_scale_shift_norm': True, 'resblock_updown': True, 'use_checkpoint': True}
        self.instantiate_first_stage(opts)
        self.model = UNetModel(**unet_config)

        # position_map = torch.load('/home/Data/tianchi/position_map.pth', map_location=self.device)
        # position_map = rearrange(position_map, 'b w h c -> b c h w')
        # self.register_buffer('position_map', position_map)
        # print('position_map', self.position_map.shape)
        self.latent_size = opts.latent_size
        self.channels = opts.latent_channel

        self.parameterization = "eps"  # all assuming fixed variance schedules
        self.loss_type = "l1"
        self.use_ema = False
        self.use_positional_encoding = False
        self.v_posterior = 0.
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        self.scale_by_std = False
        self.log_every_t = 100

        self.register_schedule(linear_start=0.00085, linear_end=0.0120)
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def instantiate_first_stage(self, opts):
        print(opts.first_stage_ckpt)
        # model = VQModelInterface.load_from_checkpoint(opts.first_stage_ckpt)
        model = VAE()
        model._scale = GLOBAL_SCALE
        # states = torch.load(opts.first_stage_ckpt, map_location=self.device)
        # model.load_state_dict(states['state_dict'])
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_first_stage(self, x):
        x = x.repeat(1, 3, 1, 1)
        h = self.first_stage_model.encode(x)
        return h

    @torch.no_grad()
    def decode_first_stage(self, z):
        xrec = self.first_stage_model.decode(z)
        xrec = torch.mean(xrec, dim=1, keepdim=True)
        return xrec

    @torch.no_grad()
    def get_input(self, batch):
        # moments = batch['latent_2D'].float()
        # guide = batch['cond1'].float()
        # label = batch['cond2'].long()
        #
        # latent = rearrange(moments, '1 c b h w -> b c h w')
        # guide = rearrange(guide, '1 c b h w -> b c h w')
        # label = rearrange(label, '1 b -> b')
        # c = [guide, label]
        target = batch['gt_data'].float()
        cond = batch['inr_data'].float()
        se_cond = batch['se_frame'].float()
        cond = torch.cat([cond, se_cond], dim=2)
        target = rearrange(target, '1 b c h w -> b c h w')
        cond = rearrange(cond, '1 b c h w -> b c h w')
        # return latent, c
        return target, cond

    def apply_model(self, x, t, cond):
        # guide, label = cond

        # print(cond.shape)
        # out = self.model(x=x, hint=guide, timesteps=t, y=label)
        out = self.model(x=x, timesteps=t, hint=cond)
        return out

    def forward(self, x, c):
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        t = t.repeat(x.shape[0])
        return self.p_losses(x, t, c)

    def p_losses(self, x_start, t, c):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # print(x_noisy.shape)
        model_out = self.apply_model(x_noisy, t, c)
        target = noise

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])
        loss = loss.mean()
        self.log('diffusion loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss


    def training_step(self, batch, batch_idx):
        z, c = self.get_input(batch)

        loss = self(z, c)

        if batch_idx == 0:
            self.sample_batch = batch
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @rank_zero_only
    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.current_epoch == 0 or self.current_epoch % self.opts.eval_save_every_n_epoch != 0:
            return
        with self.ema_scope("Plotting"):
            img_save_dir = os.path.join(self.opts.default_root_dir, 'train_progress', str(self.current_epoch))
            os.makedirs(img_save_dir, exist_ok=True)

            z, c = self.get_input(self.sample_batch)

            x_rec = self.decode_first_stage(z).to('cpu')
            x_samples = self.sample(c=c, batch_size=z.shape[0], return_intermediates=False, clip_denoised=False)
            img_samples = self.decode_first_stage(x_samples).to('cpu')

            for i in range(z.shape[0]):
                save_name = str(self.global_rank) + '_' + str(i) + '.png'
                save_image([img_samples[i] * 0.5 + 0.5], os.path.join(img_save_dir, save_name))
                save_image(x_rec[i] * 0.5 + 0.5, os.path.join(img_save_dir, 'x_rec_' + save_name))

    def configure_optimizers(self):
        base_lr = self.opts.base_lr
        accumulate_grad_batches = self.opts.accumulate_grad_batches
        batch_size = self.opts.batch_size
        devices, nodes = 1, self.trainer.num_nodes
        base_batch_size = 1
        # total_steps = self.trainer.estimated_stepping_batches
        lr = base_lr * devices * nodes * batch_size * accumulate_grad_batches / base_batch_size
        print(
            "Setting learning rate to {:.2e} = {:.2e} (base_lr) * {} (batchsize) * {} (accumulate_grad_batches) * {} (num_gpus) * {} (num_nodes) / {} (base_batch_size)".format(
                lr, base_lr, batch_size, accumulate_grad_batches, devices, nodes, base_batch_size))
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)

        scheduler_config = {'warm_up_steps': [10000], 'cycle_lengths': [10000000000000], 'f_start': [1e-06],
                            'f_max': [1.0],
                            'f_min': [1.0]}
        scheduler = LambdaLinearScheduler(**scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [
            {
                'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                'interval': 'step',
                'frequency': 1
            }]
        return [opt], scheduler


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    if opts.reproduce:
        pl.seed_everything(42, workers=True)
        opts.deterministic = True
        opts.benchmark = False
    else:
        opts.deterministic = False
        opts.benchmark = False
    if opts.command == 'fit':
        opts.default_root_dir = os.path.join(opts.result_root, opts.exp_name)
        if os.getenv("LOCAL_RANK", '0') == '0':
            if not os.path.exists(opts.default_root_dir):
                os.makedirs(opts.default_root_dir)
                # code_dir = os.path.abspath((os.getcwd()))
                # shutil.copytree(code_dir, os.path.join(opts.default_root_dir, 'code'))
                # Define paths
                # script_path = os.path.abspath(__file__)  # Path to your main script
                # project_root = os.path.abspath((os.getcwd()))  # Root of your project directory
                # dest_folder = os.path.join(opts.default_root_dir, 'code')  # Where to copy the dependencies
                #
                # # Copy script and dependencies
                # copy_dependencies(script_path, project_root, dest_folder)
                # print('save in', opts.default_root_dir)
    main(opts)
