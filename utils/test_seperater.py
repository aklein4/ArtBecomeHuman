"""make variations of input image"""

import os
from PIL import Image
import random
from tqdm import tqdm
from shutil import copyfile


# base directory to load from
BASE_INPUT_DIR = "C:/Repos/ArtBecomeHuman/data/__AI__artbench"

# list of genre sorted folders to use
GENRE_FOLDERS = [
    "art_nouveau",
    "baroque",
    "expressionism",
    "impressionism",
    "post_impressionism",
    "realism",
    "renaissance",
    "romanticism",
    "surrealism",
    "ukiyo_e"
]

# prefix to give ai files and folders
AI_PREFIX = "x"

# base output directory
BASE_OUTPUT_DIR = "C:/Repos/ArtBecomeHuman_Dataset/__AI__"

SKIP = 2

def main():

    # make the output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    set_dirs = {
        "train": os.path.join(BASE_OUTPUT_DIR, "train"),
        "val": os.path.join(BASE_OUTPUT_DIR, "validate"),
        "test": os.path.join(BASE_OUTPUT_DIR, "test"),
    }

    for s in set_dirs.values():
        os.makedirs(s, exist_ok=True)

    # iterate through sub-sets in datasets
    for genre in GENRE_FOLDERS:
        print("\n --- Folder:", genre, "---")

        # get the current input folder
        train_input_folder = os.path.join(BASE_INPUT_DIR, "train")
        train_input_folder = os.path.join(train_input_folder, genre)

        val_input_folder = os.path.join(BASE_INPUT_DIR, "test")
        val_input_folder = os.path.join(val_input_folder, genre)

        # stack of image names to convert
        train_img_list = list(os.listdir(train_input_folder))[::SKIP]
        
        val_img_list = list(os.listdir(val_input_folder))[::SKIP]

        test_count = len(val_img_list) // 10

        print("Train:", len(train_img_list), "- Val:", len(val_img_list)-test_count, "- Test:", test_count)

        genre_out_dirs = {}
        for s in set_dirs.keys():
            genre_out_dirs[s] = os.path.join(set_dirs[s], genre)

        for s in genre_out_dirs.values():
            os.makedirs(s, exist_ok=True)

        for img in tqdm(train_img_list):

            img_in_path = os.path.join(train_input_folder, img)
            img_out_path = os.path.join(genre_out_dirs["train"], AI_PREFIX+img)

            try:
                copyfile(img_in_path, img_out_path)
            except:
                pass

        for img in tqdm(val_img_list[:test_count]):

            img_in_path = os.path.join(val_input_folder, img)
            img_out_path = os.path.join(genre_out_dirs["test"], AI_PREFIX+img)            

            try:
                copyfile(img_in_path, img_out_path)
            except:
                pass

        for img in tqdm(val_img_list[test_count:]):

            img_in_path = os.path.join(val_input_folder, img)
            img_out_path = os.path.join(genre_out_dirs["val"], AI_PREFIX+img)            

            try:
                copyfile(img_in_path, img_out_path)
            except:
                pass


if __name__ == "__main__":
    main()
