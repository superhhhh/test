#coding= utf-8
import glob
import cv2
import numpy as np
import os
img_dir="/data4/liyh/Unet-master11/new_/"
mask_dir="/data4/liyh/Unet-master11/results/"
imgs = glob.glob(img_dir + "/*." + 'tif')

for imgname in imgs:
        midname = imgname[imgname.rindex("/") + 1:]
        masks = glob.glob(mask_dir + midname)
        img = cv2.imread(imgname, 0)
        mask=cv2.imread(masks[0],0)
        ret, binary = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # contours是一个列表

        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(c)  # 元組（（最小外接矩形的中心坐標），（寬，高），旋轉角度）-----> ((x, y), (w, h), θ )
        box = np.int0(cv2.boxPoints(rect))# boxpoint返回矩形 4 個頂點組成的數組
        pts1 = np.float32([box[2], box[1], box[3], box[0]])  # 左上，左下，右上，右下


        base = 1
        w = 2112
        h = 256

        if ((box[0][0] - box[2][0]) < (box[0][1] - box[2][1])):
                w = 256
                h = 2112

        pts2 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warp_mat = cv2.warpPerspective(img, M, (w, h))
        x = midname.split('.')[0]
        cv2.imwrite('/data4/liyh/Unet-master11/rect_crop/{}.tif'.format(x), warp_mat)
        #cv2.waitKey(0)