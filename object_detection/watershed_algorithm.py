#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 05:45:42 2019

@author: oussama
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    
sep_coin = cv2.imread('pennies.jpg')
display(sep_coin)

# median Blur
# Grayscale
# binary threshold
# find counters
sep_blur = cv2.medianBlur(sep_coin,25)
display(sep_blur)

gray_sep_coins = cv2.cvtColor(sep_blur,cv2.COLOR_BGR2GRAY)
display(gray_sep_coins)

ret, sep_thresh = cv2.threshold(gray_sep_coins,160,255,cv2.THRESH_BINARY_INV)
display(sep_thresh)

image,contours,hierarchy = cv2.findContours(sep_thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coin,contours,i,(255,0,0),10)

display(sep_coin)


# =============================================================================
#           watershed algorithm
# =============================================================================

img = cv2.imread('pennies.jpg')
img = cv2.medianBlur(img,35)
display(img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
display(thresh)

# Otsu's method of thresholding
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
display(thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
openeing = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
display(openeing)

sure_bg = cv2.dilate(openeing,kernel,iterations=3)
display(sure_bg)

# distance transform
dist_transform = cv2.distanceTransform(openeing,cv2.DIST_L2,5)
display(dist_transform)

ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
display(sure_fg)


sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
display(unknown)

ret, markers = cv2.connectedComponents(sure_fg,)
markers = markers + 1

markers[unknown==255] = 0
display(markers)

markers = cv2.watershed(img, markers)
display(markers)


image,contours,hierarchy = cv2.findContours(markers.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coin,contours,i,(255,0,0),10)
        
display(sep_coin)









