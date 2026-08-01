import argparse
import os
import numpy as np
import torch as th
import torch.nn.functional as F
from code.datasets import normalizer_vel, denormalizer_vel, normalizer_depth
import scipy.io as sio
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
    train_step = 20000

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

    criterion = th.nn.MSELoss()

    data_list = [1]
    for id in data_list:
        print(f'Sampling start for data {id} with batch size {args.batch_size}')
        dict = sio.loadmat(f'../dataset/seam/train/data{id}.mat')
        shot = dict['shot']
        nt, nr_ori = shot.shape
        width_gap = 1
        miss_trace = 10
        far_start = nr_ori - miss_trace
        shot_far = shot[:, far_start-args.width_size:far_start]
        real_nearshot = shot[:, -args.width_size:]

        shot_far = th.tensor(shot_far, dtype = th.float32).unsqueeze(0).unsqueeze(1).to(device=device)
        real_nearshot = th.tensor(real_nearshot, dtype = th.float32).unsqueeze(0).unsqueeze(1).to(device=device)

        shot_far = shot_far.repeat(args.batch_size, 1, 1, 1)

        for ir in range(miss_trace):
            print(f'sampling missing trace index {ir}')

            sample, _, _ = sample_fn(
                    model, shot_far,
                    (args.batch_size, args.out_channels, nt, args.width_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
            )

            shot_far = sample.clone()

        with th.no_grad():
            pred = sample.mean(dim=0, keepdim=True)
            std = sample.std(dim=0, keepdim=True)
            accs = criterion(pred[:, :, :, -miss_trace:], real_nearshot[:, :, :, -miss_trace:])

        # save file
        sio.savemat(f'{dir_output}data{id}_batch{args.batch_size}_out.mat', 
                {'predict': pred.squeeze().cpu().numpy(), 'uq':std.squeeze().cpu().numpy(), 'accs': accs.item()})

    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        use_ddim=True,
        batch_size=20,
        model_path="./checkpoints/ema_0.999_",
        width_size=32,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
