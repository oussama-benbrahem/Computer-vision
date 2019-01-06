#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 03:14:54 2019

@author: oussama
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt 

img  = cv2.imread('sammy_face.jpg')
plt.imshow(img)

edges = cv2.Canny(image=img, threshold1=127, threshold2=127)
plt.imshow(edges)

# changing threshold values
edges = cv2.Canny(image=img, threshold1=0, threshold2=255)
plt.imshow(edges)

# formula to setup threshold
med_val = np.median(img)
# lower threshold to either 0 or 70% of the median
lower = int(max(0,0.7*med_val))
# upper  threshold to either 130% of the median or the max 255
upper = int(min(255,1.3*med_val))

edges = cv2.Canny(image=img, threshold1=lower, threshold2=upper)
plt.imshow(edges)
# let's blur image first and retry

blurred_img = cv2.blur(img,ksize=(5,5))

edges = cv2.Canny(image=blurred_img, threshold1=lower, threshold2=upper)
plt.imshow(edges)



