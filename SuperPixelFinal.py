import cv2
import numpy as np
import time
from PIL import Image, ImageTk

def findAllPixelAndAverageColor(img, label, label_lsc):
    xyrgbs = []
    for x in range(256): # 取得所有label相同的像素點
        for y in range(256):
            if label_lsc[x][y] == label:
                r, g, b = img[x, y]
                xyrgbs.append((x,y,r,g,b))
    average_r, average_g, average_b = getAverageColor(xyrgbs)
    for i in range(len(xyrgbs)):
        img[xyrgbs[i][0], xyrgbs[i][1]] = average_r, average_g, average_b

def getAverageColor(xyrgbs):
    n = len(xyrgbs)
    sum_r = 0
    sum_g = 0
    sum_b = 0
    for i in range(n):
        sum_r += xyrgbs[i][2]
        sum_g += xyrgbs[i][3]
        sum_b += xyrgbs[i][4]
    average_r = sum_r // n
    average_g = sum_g // n
    average_b = sum_b // n
    return average_r, average_g, average_b
def run():
    img = cv2.imread(".imagetemp/gray_temp.jpg")
    #lsc = cv2.ximgproc.createSuperpixelLSC(img)
    lsc = cv2.ximgproc.createSuperpixelSLIC(img,region_size=9,ruler = 40.0)
    lsc.iterate(10)
    mask_lsc = lsc.getLabelContourMask()
    #print(mask_lsc)
    label_lsc = lsc.getLabels()
    #print(label_lsc)
    number_lsc = lsc.getNumberOfSuperpixels()
    #print(number_lsc)
    mask_inv_lsc =cv2.bitwise_not(mask_lsc)

    img_slic = cv2.bitwise_and(img,img,mask = mask_inv_lsc)
    #cv2.imshow("img", img_slic) ##################### show
    cv2.waitKey(0)
    #cv2.destroyWindow()

    start = time.time()
    done = []
    color = []
    for x in range(256):
        for y in range(256):
            label = label_lsc[x][y]
            if label in done:
                pass
            else:
                findAllPixelAndAverageColor(img, label, label_lsc)
                done.append(label)

    end = time.time()
    print("超像素處理完成，花費：", end - start)
    #cv2.imshow("img", img)
    #cv2.waitKey(0)
    cv2.imwrite('.imagetemp/superpixel_temp.jpg', img)

if __name__ == '__main__':
    run()