#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob


impath = list(sorted(glob.glob("dataset/im_removed/*.jpg")))
resultpath = list(sorted(glob.glob("dataset/inference_results/*.jpg")))
gtpath = list(sorted(glob.glob("dataset/gt_removed/*.png")))
result2path = list(sorted(glob.glob("dataset/val_building/*.jpg")))

print(impath[:3],
resultpath[:3],
gtpath[:3],
result2path[:3])


# In[2]:


#### calculate
import cv2
import os
import numpy as np

total_iou_maskrcnn = 0
total_iou_u2net = 0
total_precision_maskrcnn = 0
total_precision_u2net = 0
total_recall_maskrcnn = 0
total_recall_u2net = 0


for idx in range(len(impath)):
    im = cv2.imread(impath[idx])
    result = cv2.imread(resultpath[idx])
    gt = cv2.imread(gtpath[idx])
    gt = 255 - gt
    result2 = cv2.imread(result2path[idx])
    
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2GRAY)
    
    _, gt = cv2.threshold(gt, 127, 255, 0)
    _, result = cv2.threshold(result, 127, 255, 0)
    _, result2 = cv2.threshold(result2, 127, 255, 0)
    
    intersection = np.logical_and(gt, result)
    
    union = np.logical_or(gt,result)
    iou_score = np.sum(intersection) / np.sum(union)
    precision = np.sum(intersection) / len(gt[gt==255])
    total_precision_maskrcnn += precision
    recall = np.sum(intersection) / len(result[result==255])
    total_recall_maskrcnn += recall
    total_iou_maskrcnn += iou_score
    
    intersection = np.logical_and(gt, result2)
    union = np.logical_or(gt,result2)
    iou_score = np.sum(intersection) / np.sum(union)
    precision = np.sum(intersection) / len(gt[gt==255])
    total_precision_u2net += precision
    recall = np.sum(intersection) / len(result2[result2==255])
    total_recall_u2net += recall
    total_iou_u2net += iou_score
    # break

pm = total_precision_maskrcnn / len(impath)
pu = total_precision_u2net / len(impath)
rm = total_recall_maskrcnn / len(impath)
ru = total_recall_u2net / len(impath)
f1m = 2*(pm*rm)/(pm+rm)
f1u = 2*(pu*ru)/(pu+ru)
print("iou of maskrcnn:", total_iou_maskrcnn / len(impath))
print("iou of u2net:", total_iou_u2net / len(impath))
print("precision of maskrcnn:", pm)
print("precision of u2net:", pu)
print("recall of maskrcnn:", rm)
print("recall of u2net:", ru)
print("f1score of maskrcnn:", f1m)
print("f1score of u2net:", f1u)


# In[ ]:




