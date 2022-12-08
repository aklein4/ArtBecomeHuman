
import requests
import sys
import os
import json

LOAD_FOLDER = "./artbench/train/baroque"
SAVE_FILE=  "./test_prompts"

PREFIX = "https://github.com/aklein4/ArtBecomeHuman/blob/main/__AI__artbench/train/baroque/"

def main():

    for file in os.listdir(LOAD_FOLDER):
        url = PREFIX + file

        response = requests.get("https://lexica.art/api/v1/search?q=apples")
        print(response.ok)
        # j = response.json()

        # prompt = j["images"][0]["prompt"]

        # print(prompt)
        exit()




if __name__ == '__main__':
    main()