"""make variations of input image"""

import argparse, os
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


# How much to change the original image, in [0.0 (exact same), 1.0 (completely different)]
NOISE_STRENGTH = 0.25

# prompt to give the model (appended to end of genre folder name)
PROMPT = "painting of"

# base directory to load from
BASE_INPUT_DIR = "C:\\Repos\\ArtBecomeHuman\\artbench"

# dataset folders to use from base
DATA_SETS = ["train"]

# list of genre sorted folders to use
GENRE_FOLDERS = [
    "art nouveau",
    "baroque",
    "expressionism",
    "impressionism",
    "post-impressionism",
    "realism",
    "renaissance",
    "romanticism",
    "surrealism",
    "ukiyo-e"
]

# prefix to give ai files and folders
AI_PREFIX = "__AI__"

# base output directory
BASE_OUTPUT_DIR = "C:\\Repos\\ArtBecomeHuman\\"+AI_PREFIX+"artbench"

# number of images to run at a time
BATCH_SIZE = 4


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=32,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    # get options
    opt = parser.parse_args()
    seed_everything(opt.seed)

    # get config
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    # load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # this is not implemented
    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # make the output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, 256 // opt.f, 256 // opt.f], device=device)

    # iterate through datasets
    for dataset in DATA_SETS:
        print("\n ----- Dataset:", dataset, "-----")

        # dataset folder containing genre folders for input
        dataset_input_folder = os.path.join(BASE_INPUT_DIR, dataset)

        # iterate through sub-sets in datasets
        for genre in GENRE_FOLDERS:
            print("\n --- Folder:", genre, "---")

            # create dataset output folder
            dataset_output_folder = os.path.os.path.join(BASE_OUTPUT_DIR, dataset)
            os.makedirs(dataset_output_folder, exist_ok=True)
            # create genre output folder
            genre_output_folder = os.path.os.path.join(dataset_output_folder, genre)
            os.makedirs(genre_output_folder, exist_ok=True)

            # get the current input folder
            genre_input_folder = os.path.join(dataset_input_folder, genre)

            # stack of image names to convert
            img_list = list(os.listdir(genre_input_folder))
            img_list.reverse()

            img_ind = 0
            # iterate through images in sub-set
            while len(img_list) > 0:

                # name of images
                titles = []
                # prompts created by images
                prompts = []

                # create batch
                while len(prompts) < BATCH_SIZE and len(img_list) > 0:

                    # get prompt description of image
                    curr_name = img_list.pop()
                    titles.append(curr_name)
                    prompts.append(genre + " " + PROMPT + " " + curr_name.split('_')[1][:-4].replace("-", " "))

                    # verbose
                    img_ind += 1
                    print("\n -- Image", str(img_ind)+":")
                    print(prompts[-1])
                    print("")

                # run generator
                precision_scope = autocast if opt.precision == "autocast" else nullcontext
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():

                            # set up model
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(BATCH_SIZE * [""])

                            # input prompts
                            c = model.get_learned_conditioning(prompts)

                            # get samples
                            shape = [opt.C, 256 // opt.f, 256 // opt.f]
                            sample, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=BATCH_SIZE,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)

                            # get the sample
                            x_samples = model.decode_first_stage(sample)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            # save the sample
                            for x in range(len(x_samples)):
                                sample = 255. * rearrange(x_samples[x].cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(sample.astype(np.uint8)).save(
                                    os.path.join(genre_output_folder, AI_PREFIX+titles[x]))


if __name__ == "__main__":
    main()
