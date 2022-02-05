import PIL
from PIL import Image, ImageFilter
import os

def run():
    image1 = Image.open(r".imagetemp/gray_temp.jpg").convert("RGB")  # 生成的灰階圖片
    pix1 = image1.load()
    image2 = Image.open(r".imagetemp/sketch_temp.jpg").convert("RGB")  # 轉換的線稿圖片
    image2 = image2.resize((256, 256))
    #image2.show()
    pix2 = image2.load()
    image3 = Image.open(r".imagetemp/superpixel_temp.jpg").convert("RGB")  # 生成的灰階圖片轉的超像素圖片
    image3 = image3.filter(ImageFilter.SMOOTH)  # 套用平滑濾淨
    pix3 = image3.load()

    for x in range(256):
        for y in range(256):
            r1, g1, b1 = pix1[x, y]
            r2, g2, b2 = pix2[x, y]
            r3, g3, b3 = pix3[x, y]
            if 256 > r2 > 250 and 256 > g2 > 250 and 256 > b2 > 250 and r1 < 250 and g1 < 250 and b1 < 250:  # 是框線或是文字
                pix1[x, y] = r3, g3, b3
    # image1.show()

    # new_img = image1.filter(ImageFilter.MedianFilter(3))  # 使用均值濾波器
    # new_img = image1.filter(ImageFilter.SMOOTH)

    image1.save(r".imagetemp/gray_temp.jpg")  # 覆蓋掉原本的灰階圖片

if __name__ == '__main__':
    run()