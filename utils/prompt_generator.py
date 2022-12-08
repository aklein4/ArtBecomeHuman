
import requests
import sys
import os
import json

def main():


    response = requests.get("https://lexica.art/api/v1/search?q=apples")
    print(response.ok)

    exit()




if __name__ == '__main__':
    main()