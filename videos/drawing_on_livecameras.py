#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:45:45 2019

@author: oussama
"""

import cv2 

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# TOP LEFT CORNER
x = width // 2
y = height // 2

# width and height of rectangle
w = width // 4
h = height // 4

# BOTTOM RIGHT


while True:
    
    ret, frame = cap.read()
    
    cv2.rectangle(frame,(x,y), (x+w,y+h),color=(0,0,255),
                  thickness=4)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()




