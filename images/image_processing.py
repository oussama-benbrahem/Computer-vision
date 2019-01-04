
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Jan  4 10:56:47 2018

@author: oussama
"""

""" This file demonstartes image processing with open cv"""

# Color mapping

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('test_dog.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

## Blending and pasting images

# blending images of the same size 
img1 = cv2.imread('dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# resizing images
img1 = cv2.resize(img1, (1200,1200))
img2 = cv2.resize(img2, (1200,1200))

# blending
blended = cv2.addWeighted(src1=img1, alpha=0.5, 
                          src2= img2, beta=0.5,
                          gamma=0)

plt.imshow(blended)

## different size images
# overlay smalll image on top of alarger image (no blending)

img2 = cv2.resize(img2,(600,600))

large_img = img1
small_img = img2
x_offset = 0
y_offset = 0

x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]

large_img[y_offset:y_end, x_offset:x_end] = small_img

plt.imshow(large_img)


# blend together images of different sizes
img1.shape
# 1401,934,3
x_offset = 934 - 600
y_offset = 1401 - 600
img2.shape
# 600, 600, 3

roi = img1[y_offset:1401, x_offset:943]
plt.imshow(roi)

# creating mask

img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
plt.imshow(img2gray, cmap='gray')

mask_inv = cv2.bitwise_not(img2gray)
plt.imshow(mask_inv, cmap='gray')

import numpy as np
white_background = np.full(img2.shape, 255, dtype=np.uint8)
white_background.shape
bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
bk.shape

fg = cv2.bitwise_or(img2,img2, mask=mask_inv)
plt.imshow(fg)

final_roi = cv2.bitwise_or(roi, fg)
plt.imshow(final_roi)

large_img = img1
small_img = final_roi

large_img[y_offset:y_offset+small_img.shape[0], x_offset:x_offset+small_img.shape[1]] = small_img
plt.imshow(large_img)



# =============================================================================
#       image thresholding 
# =============================================================================
img1 = cv2.imread('rainbow.jpg',0)

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
plt.imshow(thresh1, cmap='gray')

img = cv2.imread('crossword.jpg',0)

def show_pic(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')

show_pic(img)

ret, th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
show_pic(th1)

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY,11,8)

show_pic(th2)

blended = cv2.addWeighted(src1=th1,alpha=0.6,src2=th2,beta=0.4,gamma=0)
show_pic(blended)

# =============================================================================
#       blurring and smoothing
# =============================================================================
def load_img():
    img = cv2.imread('bricks.jpg').astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    
i = load_img()
display_img(i)

gamma = 1/4 # image more brighter
result = np.power(i, gamma)
display_img(result)

img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,text='bricks',org=(10,600),fontFace=font,fontScale=10,
            color=(255,0,0),thickness=4)


kernel = np.ones(shape=(5,5), dtype=np.float32)/25

dst = cv2.filter2D(img,-1,kernel)
display_img(dst)

# smooth
# reset original image

blurred = cv2.blur(img,ksize=(5,5))

blurred_img = cv2.GaussianBlur(img, (5,5),10)
display_img(blurred_img)

# =============================================================================
#                       Morphological operators
# =============================================================================
def load_img():
    blank_img = np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img,text='ABCDE',org=(50,300),fontFace=font,fontScale=5,
            color=(255,255,0),thickness=4)
    return img

def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')

img = load_img()
display_img(img)

kernel = np.ones((5,5),dtype=np.uint8)
result = cv2.erode(img, kernel, iterations=1)
display_img(result)

# dilation (removing background noise)
img = load_img()
white_noise = np.random.randint(low=0,high=2,size=(600,600))
display_img(white_noise)

white_noise = white_noise * 255
display_img(white_noise)
white_noise = white_noise + img
display_img(white_noise)

opening = cv2.morphologyEx(noise_img, cv2.MORPH_OPEN,kernel)
black_noise = np.random.randint(low=0,high=2,size=(600,600))
black_noise = black_noise * -255
black_noise_img = img + black_noise
black_noise_img[black_noise_img== -255]=0
# =============================================================================
#                       GRadients: edge detection(change in colors) 
# =============================================================================



def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    
img = cv2.imread('sudoku.jpg',0)

display_img(img)
# it display verticle lines (inverse 1 and 0 to display horizontale lines)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
display_img(sobelx)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
display_img(laplacian)

sobely = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=5)
display_img(sobely)
blended = cv2.addWeighted(src1=sobelx, alpha=0.5,src2=sobely,beta=0.5,gamma=0)
display_img(blended)

ret,th1 = cv2.threshold(img, 100,125,cv2.THRESH_BINARY)
display_img(th1)

kernel = np.ones((4,4),np.uint8)
gradient = cv2.morphologyEx(blended, cv2.MORPH_GRADIENT, kernel)
display_img(gradient)


# =============================================================================
#                       Histograms (HOG)
# =============================================================================


dark_horse = cv2.imread('horse.jpg')
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)

rainbow = cv2.imread('rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

blue_bricks = cv2.imread('bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)

plt.imshow(show_horse)

hist_values = cv2.calcHist([blue_bricks], channels=[0],mask=None,histSize=[256],ranges=[0,256])

plt.plot(hist_values)

img = blue_bricks

color = ('b','g','r')

for i, col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])
    
plt.title('Histogram for blue bricks')








