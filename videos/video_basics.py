#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 10:43:54 2019

@author: oussama
"""

import cv2

capture = cv2.VideoCapture(0)


width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


writer = cv2.VideoWriter('myvideo.mp4',
                         cv2.VideoWriter_fourcc(*'XVID'),
                         20,(width,height))
while True:
    ret,frame = capture.read()
    writer.write(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
writer.release()
cv2.destroyAllWindows




