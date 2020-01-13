import cv2 as cv
import cv2 as cv
import numpy as np
import os
import xlwt
#images_dir="C:\Users\lyh\Desktop\cal"
mask_dir="C:\Users\lyh\Desktop\\book\predict"
mask_ids = next(os.walk(mask_dir))[2]
area = []
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("predict")
for index in range(len(mask_ids)):
    white_area=0
    data = []
    #img_dir = os.path.join(images_dir, image_ids[index])
    mas_dir = os.path.join(mask_dir, mask_ids[index])
    img = cv.imread(mas_dir,0)
    ret,th3=cv.threshold(img,50,255,cv.THRESH_BINARY)
    height, width = th3.shape
    for i in range(height):
        for j in range(width):
            if th3[i, j] == 255:
                white_area += 1
    data.append(mask_ids[index])
    data.append(white_area)
    area.append(data)

for i in range(0, 82):
    for j in range(0, 2):
            sheet.write(i, j, area[i][j])
workbook.save('C:\Users\lyh\Desktop\\book\data.xls')

