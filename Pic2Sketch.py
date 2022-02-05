import cv2
import os
import math
import numpy as np
import matplotlib
from PIL import Image, ImageTk

def pictosketch(image):
	img = np.array(image) # 圖片轉Array
	gray_image = img

	inverted = 255 - gray_image

	blurred = cv2.GaussianBlur(inverted, (25, 25), 0) # 高斯模糊
	blurred = cv2.GaussianBlur(blurred, (25, 25), 0) # 高斯模糊
	blurred = cv2.GaussianBlur(blurred, (25, 25), 0) # 高斯模糊
	invertedblur = 255 - blurred

	sketch = cv2.divide(gray_image, invertedblur, scale = 256.0) # 對圖片進行除法運算
	temp = Image.fromarray(sketch).resize((256, 256))
	matplotlib.image.imsave(".imagetemp/sketch_temp.jpg", temp, cmap='gray')

	return sketch # 是Array形式的圖片
