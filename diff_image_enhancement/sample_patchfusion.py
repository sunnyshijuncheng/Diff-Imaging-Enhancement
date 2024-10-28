"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.nn.functional as F

from code.datasets import _list_image_files_recursively
import scipy.io as sio
from code.train_util import parse_dataname_from_filename

from code import logger
from code.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    device = th.device('cuda')

    logger.configure()

    train_step = 400000

    if not args.use_ddim:
        dir_output = f'./output/ddpm/step{train_step}/'
        os.makedirs(dir_output, exist_ok=True)
    else:
        dir_output = f'./output/{args.timestep_respacing}/step{train_step}/'
        os.makedirs(dir_output, exist_ok=True)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(f'{args.model_path}{(train_step):06d}.pt', map_location=device)
    )
    model.to(device=device)
    model.eval()

    logger.log("sampling...")

    model_kwargs = {}

    sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    # patch setting
    patch_size = 256
    step_size = 128 # patch_size // 4

    # gaussian kernel settting
    sigma = patch_size // 8
    x = np.arange(patch_size) - patch_size // 2
    y = np.arange(patch_size) - patch_size // 2
    x, y = np.meshgrid(x, y)

    # Gaussian Weight Calculation
    gaussian_kernel = gaussian_weight(th.from_numpy(x).float(), th.from_numpy(y).float(), sigma)
    gaussian_kernel = gaussian_kernel.to(device=device)
    gaussian_kernel = gaussian_kernel.repeat(args.batch_size, 1, 1)

    # Initialize Accumulated Prediction and Weight Matrices
    accumulated_result = th.zeros((args.batch_size, nz_block, nx_block), dtype=th.float32).to(device=device)
    accumulated_weight = th.zeros((args.batch_size, nz_block, nx_block), dtype=th.float32).to(device=device)

    # load test data
    dict = sio.loadmat(f'../dataset/test/data.mat')
    sparse_np = dict['sparse']
    sparse_np = sparse_np/np.max(np.abs(sparse_np))
    nz_block, nx_block = sparse_np.shape

    for i in range(0, nz_block - patch_size + 1, step_size):
        for j in range(0, nx_block - patch_size + 1, step_size):
            # Split to patches
            sparse = sparse_np[i:i+patch_size, j:j+patch_size]
            # numpy to tensor
            sparse = th.tensor(sparse, dtype = th.float32).unsqueeze(0).unsqueeze(1).to(device=device)
            # repeat condition batch_size times
            sparse = sparse.repeat(args.batch_size times, 1, 1, 1)
          
            _, _, w, h = sparse.shape

            sample, _, _ = sample_fn(
                    model, sparse,
                    (args.batch_size, args.in_channels, w, h),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
            )

            print(i, j)

            # Accumulate Predictions
            accumulated_result[:, i:i+patch_size, j:j+patch_size] += gaussian_kernel * sample.squeeze()
            # Accumulate Weights
            accumulated_weight[:, i:i+patch_size, j:j+patch_size] += gaussian_kernel

    # Final Image Reconstruction
    final_prediction = accumulated_result / accumulated_weight

    # save final file
    sio.savemat(f'{dir_output}data_batch{args.batch_size}_out.mat', 
                    {'predict': final_prediction.cpu().numpy()})

    logger.log("sampling complete")

def gaussian_weight(x, y, sigma):
    return th.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2))).float()

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        use_ddim=True,
        batch_size=10,
        model_path="./checkpoints/ema_0.999_",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
