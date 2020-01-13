import cv2 as cv
import os
import numpy as np
images_dir="/data4/liyh/20190620-crop/20190620-crop/"
#mask_dir="/data4/liyh/Unet-master11/raw_fyc/label/"
image_ids=next(os.walk(images_dir))[2]
for i in range(len(image_ids)):
    img_dir=os.path.join(images_dir,image_ids[i])
    #masks_dir=os.path.join(mask_dir,image_ids[i])
    img=cv.imread(img_dir,0)
    x=image_ids[i].split('.')[0]
    cv.imwrite('/data4/liyh/Unet-master11/new_/{}.tif'.format(x),img)
    #cv.imwrite('/data4/liyh/Unet-master11/half/label/{}.tif'.format(x),mask)
