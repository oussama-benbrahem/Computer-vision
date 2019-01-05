#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:07:39 2019

@author: oussama
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# full image
full = cv2.imread('sammy.jpg')
full = cv2.cvtColor(full,cv2.COLOR_BGR2RGB)

# sammy face picture
face = cv2.imread('sammy_face.jpg')
face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

# example of using eval function
mystring = 'sum'
myfunc = eval(mystring)
myfunc([1,2,3])

# methods for comparaison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for m in methods:
    """ TO DO"""
    
    full_copy = full.copy()
    method = eval(m)
    res = cv2.matchTemplate(full_copy, face, method)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    height,width,channels = face.shape
    bottom_right = (top_left[0]+width,top_left[1]+height)
    cv2.rectangle(full_copy,top_left,bottom_right,(255,0,0),10)
    
    # plot and show images
    plt.subplot(121)
    plt.imshow(res)
    plt.title('heatmap of template matching')
    
    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('detecting of template')
    plt.suptitle(m)
    
    plt.show()
    print('\n')
# Best method
my_method = eval('cv2.TM_CCOEFF')
res = cv2.matchTemplate(full, face, my_method)

plt.imshow(res)
