import os
from PIL import Image
from tqdm import tqdm
from imwatermark import WatermarkDecoder
import cv2

#iterate through each image in non_ai_art
directory = "C:/Repos/data/testing_2"
print(directory)

decoder = WatermarkDecoder('bytes', 32)
marked = 0
unmarked = 0
for file in os.listdir(directory):
    #print(file)
    if file[-4:] == ".jpg":
        filepath = os.path.join(directory, file)
        
        bgr = cv2.imread(filepath)
        try:
            watermark = decoder.decode(bgr, 'dwtDct')
            print(watermark)
            marked += 1
        except:
            unmarked += 1
        #cut longer dim to shorter dim, then resize
        with Image.open(filepath) as im:
            width, height = im.size
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

            im = im.crop((left, top, right, bottom))
            im = im.resize((256, 256))
            im.save(filepath)
            # im.show()

print("program successfully completed")
print("Marked:", marked)
print("Unmarked:", unmarked)