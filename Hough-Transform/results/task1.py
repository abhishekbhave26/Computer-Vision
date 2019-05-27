# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 23:19:23 2018

@author: abhis
"""

import cv2
import numpy as np

X = cv2.imread("noise.jpg",0)
kernel =[[1,1,1],[1,1,1],[1,1,1]]
Gx=np.array(kernel,np.uint64)


def dilation(X,Gx):

    outx=[]
    s1,s2=X.shape
    outx=np.zeros([s1,s2])
    #print(outx.shape)
    s1,s2=s1-1,s2-1
    
    for i in range(0,s1):
        for j in range(0,s2):
            x=X[i][j]
            if(x==255):
                for k in range(-1,2):
                    for l in range(-1,2):
                        outx[i+k][j+l]=255
            else:
                outx[i][j]=x
            
    dil=np.array(outx,np.uint8)
    #cv2.imwrite('dil.jpg',dil)
    #cv2.imshow('Dilation',dil)
    #cv2.waitKey(0)
    return dil



def erosion(X,Gx):

    outx=[]
    rows,columns=X.shape
    outx=np.zeros([rows,columns])
    s1,s2=rows,columns
    
    for i in range(0,s1):
        for j in range(0,s2):
            d=0
            e=0
            f=0
            if X[i][j] == 255 or X[i][j] == 0:
                if (i>0) and (j>0) and (i+1<rows) and (j+1<columns):
                    
                    s=Gx[0][0]*X[i-1][j-1]+Gx[0][1]*X[i-1][j]+Gx[0][2]*X[i-1][j+1]+\
                    Gx[1][0]*X[i][j-1]+Gx[1][2]*X[i][j+1]+\
                    Gx[2][0]*X[i+1][j-1]+Gx[2][1]*X[i+1][j]+Gx[2][2]*X[i+1][j+1]
                    
                    if(s==255*8):
                        outx[i][j]=255
                    else:
                        outx[i][j]=0
            
    ero=np.array(outx,np.uint8)
    #cv2.imwrite('ero.jpg',ero)
    #cv2.imshow('Erosion',ero)
    #cv2.waitKey(0)
    return ero


print('Opening')
#opening
ero=erosion(X,Gx)
dil=dilation(ero,Gx)
cv2.imwrite('res_noise1.jpg',dil)
new=erosion(dil,Gx)
output=cv2.subtract(dil,new)
cv2.imwrite('res_bound1.jpg',output)



print('Closing')
#closing
dil=dilation(X,Gx)
ero=erosion(dil,Gx)
cv2.imwrite('res_noise2.jpg',ero)
new2=erosion(ero,Gx)
output2=cv2.subtract(ero,new2)
cv2.imwrite('res_bound2.jpg',output2)







