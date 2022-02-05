import easyocr
import cv2
from matplotlib import pyplot as plt

def getDetectionRectangle(path): # path放圖片路徑
    temp = []
    #path = "24.jpg" # 放要轉換的圖片
    reader = easyocr.Reader(['ch_sim'], recog_network='zh_sim_g2')
    result = reader.readtext(path)
    #print(result)
    img = cv2.imread(path)
    height = img.shape[0]
    width = img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for index in range(len(result)):
        top_left = tuple(result[index][0][0])
        bottom_right = tuple(result[index][0][2])
        #print(top_left, bottom_right)
        #img = cv2.rectangle(img, (int(top_left[0]),int(top_left[1])), (int(bottom_right[0]),int(bottom_right[1])) , (0,255,0), 3)
        temp.append([top_left, bottom_right])
    #plt.imshow(img)
    #plt.show()
    return width, height, temp





