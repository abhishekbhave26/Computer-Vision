# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 13:24:45 2018

@author: abhis
"""

import cv2
import numpy as np

def cursordetection():
    img=cv2.imread('C:/Users/abhis/Desktop/us/Courses/CVIP 573/homework and project/task3/pos_7.jpg')
    source=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    template = cv2.imread('template1.png',0)
    
    w=template.shape[1]
    h = template.shape[0]
    
    blur_image=cv2.GaussianBlur(source,(3,3),0)

    laplacian_image = cv2.Laplacian(blur_image,cv2.CV_64F)
    laplacian_template = cv2.Laplacian(template,cv2.CV_64F)
    
    new=np.asarray(laplacian_image,dtype=np.float32)
    new1=np.asarray(laplacian_template,dtype=np.float32)
    ssd = cv2.matchTemplate(new,new1,cv2.TM_CCOEFF_NORMED)
    threshold=0.48 
    
    loc=np.where(ssd>=threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,0,255),2 )
    
    cv2.imshow('new.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
cursordetection()   
    








