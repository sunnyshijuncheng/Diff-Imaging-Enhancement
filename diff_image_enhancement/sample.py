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

    # use the model training at which step 
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

    model_kwargs = {}
    if args.class_cond:
        classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes

    # define sample (DDIM or DDPM)
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    dict = sio.loadmat(f'../dataset/test/data.mat')
    sparse = dict['sparse']

    sparse = th.tensor(sparse, dtype = th.float32).unsqueeze(0).unsqueeze(1).to(device=device)

    # repeat condition batch_size times
    sparse = sparse.repeat(args.batch_size, 1, 1, 1)

    _, _, w, h = sparse.shape

    # start sampling
    logger.log("sampling...")
    sample, sample_all, pred_xstart = sample_fn(
            model, sparse,
            (args.batch_size, args.in_channels, w, h),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
    )

    sample_all = th.stack(sample_all)
    pred_xstart = th.stack(pred_xstart)

    # save file
    sio.savemat(f'{dir_output}data_batch{args.batch_size}_out.mat', 
                {'predict': sample.squeeze().cpu().numpy(),
                 'sample_all': sample_all.squeeze().cpu().numpy(),
                 'pred_xstart': pred_xstart.squeeze().cpu().numpy()})

    logger.log("sampling complete")


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
