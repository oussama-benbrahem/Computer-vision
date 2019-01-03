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
img.shape
# Wrong path
img = cv2.imread('wrongpath.png')



# matplotlib --> RGB 
# Opencv --> BGR
plt.imshow(img)

# Transforming RGB to BGR
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)


img_gray = cv2.imread('test_dog.jpg', cv2.IMREAD_GRAYSCALE)
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


# =============================================================================
#                   Drawing on images
# =============================================================================
import cv2 
import numpy as np
import matplotlib.pyplot as plt

blank_img = np.zeros(shape=(512,512,3), dtype=np.int16)

blank_img.shape
plt.imshow(blank_img)

# drawing a blank rectangle
cv2.rectangle(blank_img, pt1=(384,0), pt2=(510,150),
              color=(0,255,0),thickness=10)

plt.imshow(blank_img)

# drawing a blank circle
cv2.circle(img=blank_img, center=(100,100), radius=50,
              color=(255,0,0),thickness=10)
plt.imshow(blank_img)

# drawing a line
cv2.line(blank_img, pt1=(0,0), pt2=(512,512), color=(102,255,255), thickness=5)
plt.imshow(blank_img)

# write on images
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank_img,text='Hello', org=(10,500), fontFace=font, fontScale=4,
             color=(255,255,255), thickness=3, lineType=cv2.LINE_AA)
plt.imshow(blank_img)

# Draw polygon
new_img = np.zeros(shape=(512,512,3), dtype=np.int32)

vertices = np.array([[100,300], [200,200], [400,300], [200,400]], dtype=np.int32)

pts = vertices.reshape((-1,1,2))

cv2.polylines(new_img, [pts], isClosed=True, color=(255,0,0), thickness=5)
plt.imshow(new_img)



# =============================================================================
#       drawing on images with the mouse
# =============================================================================
import cv2
import numpy as np


# function
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 100, (0,255,0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
        

cv2.namedWindow(winname='my_drawing')
cv2.setMouseCallback('my_drawing', draw_circle)

img = np.zeros((512,512,3), dtype=np.int8)
# showing image with open cv
while True:
    cv2.imshow('my_drawing', img)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()

###########################################################
# variables

# True while mouse button down
drawing = False
ix, iy = -1, -1

# Function
def draw_rectangle(event, x, y, flags, params):
    global ix, iy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
    




# showing the image

img = np.zeros((512, 512,3))
cv2.namedWindow(winname= "my_drawing")
cv2.setMouseCallback("my_drawing", draw_rectangle)

while True:
    cv2.imshow('my_drawing', img)
    
    # CHECKS for esc
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()














