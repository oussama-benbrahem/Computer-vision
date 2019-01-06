#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 03:59:51 2019

@author: oussama
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('internal_external.png')
plt.imshow(img,cmap='gray')

image,contours,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

external_contours = np.zeros(image.shape)
# draw contours
for i in range(len(contours)):
    
    #EXTERNAL
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours,contours,i,255,-1)
        
plt.imshow(external_contours,cmap='gray')

#### internal contours

internal_contours = np.zeros(image.shape)
# draw contours
for i in range(len(contours)):
    
    #EXTERNAL
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(internal_contours,contours,i,255,-1)
        
plt.imshow(internal_contours,cmap='gray')






