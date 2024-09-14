"""
Train a diffusion model
"""
import argparse

from code import logger
from code.datasets import load_data
from code.resample import create_named_schedule_sampler
from code.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from code.train_util import TrainLoop
import torch as th

def main():
    # define args
    args = create_argparser().parse_args()

    logger.configure()

    # define device
    device = th.device('cuda')

    logger.log("creating model and diffusion...")

    # define model and diffusion model 
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(device)

    # define sampler
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    # load data
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="../dataset/train/",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
