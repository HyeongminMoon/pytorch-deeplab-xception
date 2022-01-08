#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob


impath = list(sorted(glob.glob("dataset/im_removed/*.jpg")))
resultpath = list(sorted(glob.glob("dataset/inference_results/*.jpg")))
gtpath = list(sorted(glob.glob("dataset/gt_removed/*.png")))
result2path = list(sorted(glob.glob("dataset/val_building/*.jpg")))

print(impath[:3],
resultpath[:3],
gtpath[:3],
result2path[:3])

print(len(impath))


# In[3]:


import cv2
import os

for idx in range(len(impath)):
    im = cv2.imread(impath[idx])
    result = cv2.imread(resultpath[idx])
    gt = cv2.imread(gtpath[idx])
    gt = 255 - gt
    result2 = cv2.imread(result2path[idx])
    
    img_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img_gray, 127, 255, 0)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        cv2.drawContours(result, [cnt], 0, (0, 0, 255), 3)
        cv2.drawContours(result2, [cnt], 0, (0, 0, 255), 3)
    
    # plt.imshow(cv2.hconcat([im, result]))
    # plt.show()
    # plt.imshow(cv2.hconcat([im, result2]))
    # plt.show()
    os.makedirs('maskrcnn/',exist_ok=True)
    os.makedirs('u2net/',exist_ok=True)
    cv2.imwrite("maskrcnn/"+os.path.basename(gtpath[idx]),cv2.hconcat([im, result]))
    cv2.imwrite("u2net/"+os.path.basename(gtpath[idx]),cv2.hconcat([im, result2]))






