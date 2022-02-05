import textDetection
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def RGBGapLess5(r, g, b): # 判斷之間的差值是否小於15
    if abs(r-g) < 15 and abs(r-b) < 15 and abs(g-b) < 15:
        #print("True", r, g, b)
        return True
    else:
        #print("False",r, g, b)
        return False

def resizeDetectionRectangle(detectionRectangle, width, height): # 將線稿偵測到的長方形區域縮放成上色後圖片的對應大小
    temp = []
    for i in range(len(detectionRectangle)):
        x1 = int(detectionRectangle[i][0][0]*256/width)
        y1 = int(detectionRectangle[i][0][1]*394/height)
        x2 = int(detectionRectangle[i][1][0]*256/width)
        y2 = int(detectionRectangle[i][1][1]*394/height)
        temp.append([(x1, y1), (x2, y2)])
    return temp

def run():
    sketchPath = r".imagetemp\sketch_temp.jpg"  # 線稿圖片
    colorPath = r".imagetemp\color_temp.jpg"  # 上色後的圖片
    originalPath = r".imagetemp\original_temp.jpg" # 原始輸入圖片
    width, height, detectionRectangle = textDetection.getDetectionRectangle(sketchPath)
    #print(width," ",  height)
    # print(detectionRectangle)
    detectionRectangle = resizeDetectionRectangle(detectionRectangle, width, height)
    # print(detectionRectangle)
    colorImg = cv2.imread(colorPath)
    # colorImg = cv2.resize(colorImg, (256, 394))
    colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
    # sketchImg = cv2.imread(sketchPath)
    # sketchImg = cv2.cvtColor(sketchImg, cv2.COLOR_BGR2RGB)
    # sketchImg = cv2.resize(sketchImg, (256,394))
    originalImg = cv2.imread(originalPath)
    originalImg = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)
    kernel = np.ones((2, 2), np.uint8)
    # originalImg = cv2.erode(originalImg, kernel, iterations=1)
    originalImg = cv2.resize(originalImg, (256, 394))
    #plt.imshow(colorImg)
    #plt.show()
    #print(detectionRectangle)
    for i in range(len(detectionRectangle)):
        colorImg = cv2.rectangle(colorImg, (detectionRectangle[i][0][0], detectionRectangle[i][0][1]),
                                 (detectionRectangle[i][1][0], detectionRectangle[i][1][1]), (0, 255, 0), 1)
        for j in range(detectionRectangle[i][0][0], detectionRectangle[i][1][0], 1):
            # print(detectionRectangle[i][0][1], detectionRectangle[i][1][1])
            for k in range(detectionRectangle[i][0][1], detectionRectangle[i][1][1], 1):
                try:
                    r, g, b = colorImg[k, j]  # h, w
                    if (RGBGapLess5(int(r), int(g), int(b))):
                        colorImg[k, j] = originalImg[k, j]
                except:
                    pass

    finalImg = colorImg
    plt.imsave(".imagetemp/color_temp.jpg", finalImg)
    print("文字偵測處理完成")

if __name__ == '__main__':
    run()




