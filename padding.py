#coding= utf-8
import cv2 as cv
import numpy as np
import os

'''

Fill the image to a multiple of 32

'''

images_dir='/data4/liyh/Unet-master11/new_/'
label_dir='/data4/liyh/Unet-master11/raw/label/'
images_ids=next(os.walk(images_dir))[2]
for i in range(len(images_ids)):
    img_dir=os.path.join(images_dir,images_ids[i])
    #mask_dir=os.path.join(label_dir,images_ids[i])
    img=cv.imread(img_dir,0)
    #mask=cv.imread(mask_dir,0)
    image=np.copy(img)
    #masks=np.copy(mask)
    h,w=np.shape(img)[:2]
    if h%32!=0:
        top_pad=(32-h%32)//2
        bottom_pad=32-h%32-top_pad
    else:
        top_pad=bottom_pad=0
    if w%32!=0:
        left_pad = (32-w%32)//2
        right_pad = 32-w%32-left_pad
    else:
        left_pad=right_pad=0
    padding=[(top_pad,bottom_pad),(left_pad,right_pad)]
    img_padding = np.pad(image,padding,mode='constant',constant_values=0)
    #label_padding = np.pad(masks,padding,mode='constant',constant_values=0)
    name=images_ids[i].split(".")[0]
    img_name=os.path.join(  '/data4/liyh/Unet-master11/test/','{}.tif'.format(name))
    #label_name=os.path.join('/data4/liyh/Unet-master11/test/','{}.tif'.format(name))
    cv.imwrite(img_name,img_padding)
    #cv.imwrite(label_name,label_padding)
