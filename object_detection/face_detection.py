#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 11:50:26 2019

@author: oussama
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

nadia = cv2.imread('Nadia_Murad.jpg',0)
denis = cv2.imread('Denis_Mukwege.jpg',0)
solvay = cv2.imread('solvay_conference.jpg',0)

plt.imshow(solvay,cmap='gray')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
        
    return face_img

result = detect_face(solvay)
plt.imshow(result,cmap='gray')


# adjust face detection
def adj_detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,
                                               minNeighbors=5)
    
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
        
    return face_img

result = adj_detect_face(solvay)
plt.imshow(result,cmap='gray')

# Eye detector
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_eyes(img):
    face_img = img.copy()
    eyes_rects = eye_cascade.detectMultiScale(face_img,scaleFactor=1.2,
                                               minNeighbors=5)
    
    for (x,y,w,h) in eyes_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
        
    return face_img

result = detect_eyes(nadia)
plt.imshow(result,cmap='gray')

cap = cv2.VideoCapture(0)
    
while True:
    ret,frame = cap.read(0)
    frame = detect_face(frame)
    
    cv2.imshow('video face detect', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
