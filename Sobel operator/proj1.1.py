# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:51:13 2018

@author: abhis
"""

import cv2
import numpy as np

img = cv2.imread("C:/Users/abhis/Desktop/us/Courses/CVIP 573/homework and project/task1.png", 0)

height = img.shape[0]
width = img.shape[1]

#padding
outimg=[]
for i in range(height+2):
        rowList=[]
        for j in range(width+2):
            rowList.append(0)
        outimg.append(rowList)

for i in range(0,height):
        for j in range(0,width):
            outimg[i+1][j+1]=img[i][j]

x=np.array(outimg,dtype=np.uint8)

#flipped sobel operator
Gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float)

#flipped sobel operator
Gy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float)

#convolution function
def convolve(X,Gx,Gy):
    rows=X.shape[0]
    columns=X.shape[1]
    
    outx=[]
    outy=[]
    for i in range(rows):
        rowList=[]
        for j in range(columns):
            rowList.append(0)
        outx.append(rowList)
    
    for i in range(rows):
        rowList=[]
        for j in range(columns):
            rowList.append(0)
        outy.append(rowList)

    for i in range(1,rows-1):
        for j in range(1,columns-1):
            
            s=Gx[0][0]*X[i-1][j-1]+Gx[0][1]*X[i-1][j]+Gx[0][2]*X[i-1][j+1]+\
                Gx[1][0]*X[i][j-1]+Gx[1][1]*X[i][j]+Gx[1][2]*X[i][j+1]+\
                Gx[2][0]*X[i+1][j-1]+Gx[2][1]*X[i+1][j]+Gx[2][2]*X[i+1][j+1]
            
            p=Gy[0][0]*X[i-1][j-1]+Gy[0][1]*X[i-1][j]+Gy[0][2]*X[i-1][j+1]+\
                Gy[1][0]*X[i][j-1]+Gy[1][1]*X[i][j]+Gy[1][2]*X[i][j+1]+\
                Gy[2][0]*X[i+1][j-1]+Gy[2][1]*X[i+1][j]+Gy[2][2]*X[i+1][j+1]
                        
            outx[i-1][j-1]=s
            outy[i-1][j-1]=p
    
    #eliminate zero values        
    x_min=outx[0][0]
    x_max=outx[0][0]
    for i in range(0,len(outx)):
        for j in range(0,i):
            if(x_min>outx[i][j]):
                x_min=outx[i][j]
            if(x_max<outx[i][j]):
                x_max=outx[i][j]
    
    y_min=outy[0][0]
    y_max=outy[0][0]            
    for i in range(0,len(outy)):
        for j in range(0,i):
            if(y_min>outy[i][j]):
                y_min=outy[i][j]
            if(y_max<outy[i][j]):
                y_max=outy[i][j]
                
    pos_edge_x = (outx- x_min) / (x_max - x_min)
    pos_edge_y = (outy- y_min) / (y_max - y_min)
    
    return pos_edge_x,pos_edge_y
   
imgx,imgy=convolve(x,Gx,Gy)         
        
cv2.imshow('hori.png',imgx)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('verti.png',imgy)
cv2.waitKey(0)
cv2.destroyAllWindows()