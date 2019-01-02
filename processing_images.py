#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 13:25:40 2018

@author: oussama
"""

""" Preprocessing images with open cv"""


import numpy as np
import matplotlib.pyplot as plt

import cv2

# Reading image
img = cv2.imread('test_dog.jpg')
# image type
type(img)
# Wrong path
img = cv2.imread('wrongpath.png')

img.shape

# matplotlib --> RGB 
# Opencv --> BGR
plt.imshow(img)

# Transforming RGB to BGR
fix_img = cv2.cv2Color(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)


img_gray = cv2.imread('test_dog.png', cv2.IMREAD_GRAYSCALE)
# 2D because of grayscale
img_gray.shape
# showing grayscale images
# cmap: gray, magma
plt.imshow(img_gray, cmap='gray')

# Resizing images
new_img = cv2.resize(fix_img, (1000, 400))
plt.imshow(new_img)

w_ratio = 0.5
h_ratio = 0.5

new_img = cv2.resize(fix_img, (0,0), fix_img, w_ratio, h_ratio)
plt.imshow(new_img)

new_img.shape

# image flipping

new_img = cv2.flip(fix_img, -1)
plt.imshow(new_img)

# save files
cv2.imwrite('test.png', fix_img)


# Display image larger smaller
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.imshow(fix_img)


#














