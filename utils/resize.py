import os
from PIL import Image

#iterate through each image in non_ai_art
directory = os.getcwd() + '\\non_ai_art'
print(directory)
for file in os.listdir(directory):
    #print(file)
    if file.endswith('jpg'):
        filepath = os.path.join(directory, file)
        #cut longer dim to shorter dim, then resize
        with Image.open(directory+'/'+file) as im:
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