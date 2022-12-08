"""make variations of input image"""
import requests
import shutil
import csv
import os
import time

CSV_PATH = "ArtBecomeHuman/ArtBench-10.csv" 

# base directory to load from
BASE_INPUT_DIR = "artbench"

# dataset folders to use from base
DATA_SETS = ["train"]

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
BASE_OUTPUT_DIR = "ai_images"

#for reverse image search
LEXICA_PREFIX = "https://lexica.art/api/v1/search?q="

#Indices for csv file
NAME_IND = 0
URL_IND = 2
GENRE_IND = 6


# Get JSON response from lexica, add check that response is valid
def get_lexica_dict(img_url):
    response = requests.get(LEXICA_PREFIX + img_url)
    print(response.status_code)
    j = response.json()
    return j["images"]
    

# Download image from Lexica
def download_image(src, file_sig, output_folder):
    filename = output_folder + file_sig
    res = requests.get(src, stream = True)
    if res.status_code == 200:
        with open(filename, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
    else:
        print("Fuck me")


def main():
    # iterate through datasets
    for dataset in DATA_SETS:

        print("\n ----- Dataset:", dataset, "-----")

        # dataset folder containing genre folders for input
        dataset_input_folder = os.path.join(BASE_INPUT_DIR, dataset)

        # keeps track of url, name, and genre for proper downloading
        genre_dict = {}
        
        # iterate through sub-sets in datasets, this creates folders to place images
        for genre in GENRE_FOLDERS:
           # print("\n --- Folder:", genre, "---")
            # create dataset output folder
            dataset_output_folder = os.path.os.path.join(BASE_OUTPUT_DIR, dataset)
            os.makedirs(dataset_output_folder, exist_ok=True)
            # create genre output folder
            genre_output_folder = os.path.os.path.join(dataset_output_folder, genre)
            os.makedirs(genre_output_folder, exist_ok=True)

            # get the current input folder
            genre_input_folder = os.path.join(dataset_input_folder, genre)
            genre_dict[genre] = genre_input_folder

            
        with open(CSV_PATH, 'r') as f:
            cr = csv.reader(f)
            iter = 0
            for row in cr:
                genre = row[GENRE_IND]
                if genre == 'label':
                    continue
                print(genre + str(iter))
                url = row[URL_IND]
                name = row[NAME_IND]
                lex_dic = get_lexica_dict(url)
                #wait for .25 seconds
                time.sleep(0.25)

                download_image(lex_dic[iter]["src"], AI_PREFIX + name, genre_dict[genre])
                #wait for .25 seconds
                time.sleep(0.25)
                iter += 1

if __name__ == "__main__":
    main()