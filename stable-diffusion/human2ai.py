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


# How much to change the original image, in [0.0 (exact same), 1.0 (completely different)]
NOISE_STRENGTH = 0.5

# prompt to give the model (appended to end of genre folder name)
PROMPT = "painting of"

# base directory to load from
BASE_INPUT_DIR = "C:\\Repos\\ArtBecomeHuman\\artbench"

# dataset folders to use from base
DATA_SETS = ["test", "train"]

# list of genre sorted folders to use
GENRE_FOLDERS = [
    "romanticism",
    "art nouveau",
    "baroque",
    "expressionism",
    "impressionism",
    "post-impressionism",
    "realism",
    "renaissance",
    "surrealism",
    "ukiyo-e"
]

# prefix to give ai files and folders
AI_PREFIX = "__AI__"

# base output directory
BASE_OUTPUT_DIR = "C:\\Repos\\ArtBecomeHuman\\"+AI_PREFIX+"artbench"


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
        default=200,
        help="number of ddim sampling steps",
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
        default=5.0,
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
    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # make the output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # batch size is one image per
    batch_size = 1

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

            # iterate through images in sub-set
            img_ind = 0
            for curr_img in os.listdir(genre_input_folder):
                img_ind += 1

                # get image path
                init_img = os.path.join(genre_input_folder, curr_img)

                # load the image
                assert os.path.isfile(init_img)
                init_image = load_img(init_img).to(device)

                # get prompt description of image
                curr_prompt = genre + " " + PROMPT + " " + curr_img.split('_')[1][:-4].replace("-", " ")

                print("\n -- Image", str(img_ind)+":")
                print(curr_prompt)
                print("")

                # init image?
                init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

                # init sampler
                sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

                # set noise and sampling steps
                assert 0. <= NOISE_STRENGTH <= 1., 'can only work with strength in [0.0, 1.0]'
                t_enc = int(NOISE_STRENGTH * opt.ddim_steps)

                # run generator
                precision_scope = autocast if opt.precision == "autocast" else nullcontext
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():

                            # get conditioning
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            c = model.get_learned_conditioning([curr_prompt])

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,)

                            # get the sample
                            x_samples = model.decode_first_stage(samples)
                            sample = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)[0]

                            # save the sample
                            sample = 255. * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(sample.astype(np.uint8)).save(
                                os.path.join(genre_output_folder, AI_PREFIX+str(curr_img)))


if __name__ == "__main__":
    main()
