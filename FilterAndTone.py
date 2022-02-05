from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import math
import cv2
#打開圖片
#img = Image.open("1.jpg")
#im.show()
def filter1(img): # 灰階
    im_width, im_height = img.size
    pixels = img.load()
    for i in range(im_width):
        for j in range(im_height):
            r, g, b = pixels[i, j]
            gray = math.floor((r*30 + g*59 + b*11 + 50)/100)
            pixels[i,j] = (gray, gray, gray)
    # img.show()
    #img.save("gray.jpg")
    return img
#filter1(img)

def filter2(img): # 平滑
    img = img.filter(ImageFilter.SMOOTH_MORE)
    # img.show()
    #img.save("smooth.jpg")
    return img
#filter2(img)

def filter3(img): # 銳化
    img = img.filter(ImageFilter.SHARPEN)
    # img.show()
    #img.save("sharp.jpg")
    return img
#filter3(img)

def filter4(img): # 光照
    img_width, img_height = img.size
    pixels = img.load()
    for i in range(img_width):
        for j in range(img_height):
            r, g, b = pixels[i, j]
            r = 250 if r+30>250 else r+45
            g = 250 if g+30>250 else g+45
            b = 250 if b+30>250 else b+45
            pixels[i,j] = (r, g, b)
    # img.show()
    #img.save("softlight.jpg")
    return img
#filter4(img)

def filter5(img): # 歲月
    im_width, im_height = img.size
    pixels = img.load()
    for i in range(im_width):
        for j in range(im_height):
            r, g, b = pixels[i, j]
            b = int(math.sqrt(b)*12)
            pixels[i, j] = r, g, b
    # img.show()
    #img.save("years.jpg")
    return img
#filter5(img)

def filter6(img): # 對比
    filter = ImageEnhance.Contrast(img)
    img = filter.enhance(5)
    # img.show()
    #img.save("saturation.jpg")
    return img
#filter6(img)

def tone1(img): # 暖色調
    # img.show()
    img_width, img_height = img.size
    pixels = img.load()
    for i in range(img_width):
        for j in range(img_height):
            r, g, b = pixels[i, j]
            b = 15 if (g < 60 and r < 60) else b
            g = 250 if g + 30 > 250 else g + 45
            r = 250 if r + 30 > 250 else r + 45
            pixels[i, j] = (r, g, b)
    # img.show()
    #img.save("removeRED.jpg")
    return img
#tone1(img)

def tone2(img): # 冷色調
    # img.show()
    img_width, img_height = img.size
    pixels = img.load()
    for i in range(img_width):
        for j in range(img_height):
            r, g, b = pixels[i, j]
            r = 35 if (g < 60 and b < 60) else r
            g = 250 if g + 30 > 250 else g + 45
            b = 250 if b + 30 > 250 else b + 45
            pixels[i, j] = (r, g, b)
    # img.show()
    #img.save("removeRED.jpg")
    return img
#tone2(img)

def run(useGrayscaleFileter,
        useSmoothFileter,
        useShapeFileter,
        useLightFileter,
        useYearsFileter,
        useConstractFileter,
        useWarmTone,
        useColdTone):
    img = Image.open(".imagetemp/color_temp.jpg")

    if useGrayscaleFileter:
        img = filter1(img)
    elif useSmoothFileter:
        img = filter2(img)
    elif useShapeFileter:
        img = filter3(img)
    elif useLightFileter:
        img = filter4(img)
    elif useYearsFileter:
        img = filter5(img)
    elif useConstractFileter:
        img = filter6(img)

    if useWarmTone:
        img = tone1(img)
    elif useColdTone:
        img = tone2(img)

    img.save(".imagetemp/color_temp_new.jpg")
if __name__ == "__main__":
    pass



