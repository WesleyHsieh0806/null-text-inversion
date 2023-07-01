import os 
import json
import shutil
import argparse
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc

from null_text_inversion import ptp_utils, seq_aligner, img_utils
from null_text_inversion import load_512, NullInversion, text2image_ldm, text2image_ldm_stable, run_and_display, AttentionStore


from torch.optim.adam import Adam
from PIL import Image

def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="path to image name",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="path to output original image, reconstruction, and inverted image",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="conditional text prompt",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help='# ddim denoising steps'
    )
    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help='height of the reconstructed image'
    )
    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help='width of the reconstructed image'
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help='classifier-free guidance scale'
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='which cuda device to use'
    )

    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="",
        help='path to model checkpoint'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    MY_TOKEN = ''
    LOW_RESOURCE = False 
    NUM_DDIM_STEPS = args.ddim_steps
    GUIDANCE_SCALE = args.guidance_scale
    MAX_NUM_WORDS = 77
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    # create model
    model_path = args.model_checkpoint if args.model_checkpoint else "stabilityai/stable-diffusion-2-base"
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer


    # create NULL Inversion instance
    null_inversion = NullInversion(ldm_stable, NUM_DDIM_STEPS)


    image_path = args.image
    prompt = args.prompt
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, args.H, args.W, offsets=(0,0,0,0), verbose=True)

    print("Modify or remove offsets according to your image!")

    # Get inverted image
    prompts = [prompt]
    controller = AttentionStore()
    image_inv, x_t = run_and_display(ldm_stable, prompts, controller, NUM_DDIM_STEPS, args.H, args.W, guidance_scale=GUIDANCE_SCALE, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, verbose=False)

    # save the image
    img_utils.save_images([image_gt, image_enc, image_inv[0]], args.output_dir)

    # save unconditional embeddings
    uncond_embeddings = {
        i: uncond_embeddings[i] for i in range(len(uncond_embeddings))
    }
    torch.save(uncond_embeddings, os.path.join(args.output_dir, "uncond_embeddings.pth"))

    # save args as config
    with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
        # Save args as json
        json.dump(args.__dict__, f, indent=2)
