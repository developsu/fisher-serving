


import os
import cv2
import glob

img_list = glob.glob('./resources/samples/*')

for i in range(len(img_list)):
    img = cv2.imread(img_list[i])
    img_resize = cv2.resize(img, dsize=(0,0), fx=0.3, fy=0.3)
    filename = os.path.basename(img_list[i])
    cv2.imwrite(f'./resources/samples/resized_{filename}', img_resize)