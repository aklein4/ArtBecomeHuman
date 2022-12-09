"""make variations of input image"""

import os
from PIL import Image
import random
from tqdm import tqdm


# base directory to load from
BASE_INPUT_DIR = "C:\Repos\manual_imgs"

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
AI_PREFIX = "__AI__"

# base output directory
BASE_OUTPUT_DIR = "C:\Repos\ArtBecomeHuman_Dataset\__AI__"

IMAGE_SIZE = 256

def main():

    # make the output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    set_dirs = {
        "train": os.path.join(BASE_OUTPUT_DIR, "train"),
        "validate": os.path.join(BASE_OUTPUT_DIR, "validate"),
        "test": os.path.join(BASE_OUTPUT_DIR, "test"),
    }

    for s in set_dirs.values():
        os.makedirs(s, exist_ok=True)

    # iterate through sub-sets in datasets
    for genre in GENRE_FOLDERS:
        print("\n --- Folder:", genre, "---")

        # get the current input folder
        genre_input_folder = os.path.join(BASE_INPUT_DIR, genre)

        # stack of image names to convert
        full_img_list = list(os.listdir(genre_input_folder))
        
        random.shuffle(full_img_list)

        val_count = len(full_img_list) // 5
        test_count = val_count // 10
        val_count -= test_count

        print("Train:", len(full_img_list)-test_count-val_count, "- Val:", val_count, "- Test:", test_count)

        img_lists = {
            "train": full_img_list[test_count+val_count:],
            "validate": full_img_list[:val_count],
            "test": full_img_list[val_count:val_count + test_count]
        }

        for s in img_lists.keys():
            imgs = img_lists[s]

            output_dir = os.path.join(set_dirs[s], genre)
            os.makedirs(output_dir, exist_ok=True)

            for img in tqdm(imgs):
                if img[-4:] != ".jpg":
                    continue

                img_in_path = os.path.join(genre_input_folder, img)
                img_out_path = os.path.join(output_dir, AI_PREFIX+img)

                with Image.open(img_in_path) as image:
                    width, height = image.size

                    if (width < height):
                        left = 0
                        top = (height - width)/2
                        right = width
                        bottom = (height + width)/2
                    else:
                        left = (width - height)/2
                        top = 0
                        right = (width + height)/2
                        bottom = height

                    image = image.crop((left, top, right, bottom))
                    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
                    image.save(img_out_path)




            


if __name__ == "__main__":
    main()
