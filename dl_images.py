"""make variations of input image"""
import requests
import shutil
import csv
import time
import pandas as pd

HOME_PATH = "//home//matthewvpersonal22"
CSV_PATH = HOME_PATH + "//interet_art_catalog.csv"

# base directory to load from

# dataset folders to use from base

FILE_HEADER = "__HUMAN_GoA__"
FILE_TYPE = ".jpg"

OUTPUT_DIR = HOME_PATH + "/art_images/"
URL_HEADER = "https://www.wga.hu/detail/"
# Indices for csv file
NAME_IND = 2
URL_IND = 6
FORM_IND = 7


# Get JSON response from lexica, add check that response is valid

# Download image from Lexica
def download_image(src, file_sig, output_folder):
    filename = output_folder + file_sig
    res = requests.get(src, stream=True)
    if res.status_code == 200:
        with open(filename, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
    else:
        print("Error at file with source: " + src)


def main():
    painting_url_list = []
    cr = pd.read_csv(CSV_PATH, encoding='unicode_escape')
    itr = 0
    size = cr.shape[0]
    for i in range(size):
        itr += 1
        if itr == 0:
            continue
        if itr % 100 == 0:
            print("Load Progress: " + str(itr) + " out of " + str(size))
        if cr.iat[i, FORM_IND] == 'painting':
            st_ind = cr.iat[i, URL_IND].find("/html/") + 6
            painting_url_list.append((cr.iat[i, NAME_IND] + str(itr), URL_HEADER + cr.iat[i,URL_IND][st_ind:-4] + "jpg"))
    num_to_dl = len(painting_url_list)
    print("Loading Complete")
   # start = time.time()
    itr = 0
    for painting in painting_url_list:
        itr += 1
        if itr % 100 == 0:
            print("Download Progress: " + str(itr) + " out of " +  str(num_to_dl))
            time.sleep(0.25)
        download_image(painting[1], FILE_HEADER + painting[0].replace(" ", "") + FILE_TYPE, OUTPUT_DIR)
    #end = time.time()
   # print("Time Elapsed for Downloads: " + str(end - start))
    
if __name__ == "__main__":
    main()
