# 對圖片進行縮放
from PIL import Image
import os

def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_name):
        #print(filename) #just for test
        #img is used to store the image data
        img = Image.open(directory_name + "/" + filename)
        new_img = img.resize((256, 394))
        # new_img.show()
        path = r'C:\Users\User\Desktop\備審資料\專題報告'
        if not os.path.isdir(path):
            os.mkdir(path)
        #new_img.save(path + "/" + filename + "_new" + ".jpg")
        new_img.save(path + "/" + filename)

if __name__ == '__main__':
    fileRoot = r"C:\Users\User\Desktop\備審資料\專題報告"
    read_directory(fileRoot)

