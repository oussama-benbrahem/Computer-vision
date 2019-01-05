#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:38:03 2019

@author: oussama
"""

import cv2
import time

cap = cv2.VideoCapture('hand_move.mp4')

if cap.isOpened() == False:
    print("file not found !")
    
while cap.isOpened():
    
    ret, frame = cap.read()
    
    if ret == True:
        time.sleep(1/20)
        cv2.imshow("frame",frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()
    
