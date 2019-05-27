# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 23:19:26 2018

@author: abhis
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

imgnew = cv2.imread("point.jpg",0)
#mention in report that I used different image 
img = cv2.imread('turbine-blade.jpg',0)
imgnew=cv2.imread('turbine-blade.jpg')
img2= cv2.imread("segment.jpg",0)


def calc(X,lthreshold):
    rows,columns=X.shape
    outx=np.zeros([rows,columns],np.uint8)
    kernel =[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    Gx=np.array(kernel)
    
    for i in range(1,rows-1):
        for j in range(1,columns-1):
                
            s=Gx[0][0]*X[i-1][j-1]+Gx[0][1]*X[i-1][j]+Gx[0][2]*X[i-1][j+1]+\
                Gx[1][0]*X[i][j-1]+Gx[1][1]*X[i][j]+Gx[1][2]*X[i][j+1]+\
                Gx[2][0]*X[i+1][j-1]+Gx[2][1]*X[i+1][j]+Gx[2][2]*X[i+1][j+1]
            
            if(s>lthreshold):            
                outx[i-1][j-1]=255
            
    return outx


def segment(X,threshold):
    
    rows,columns=X.shape
    outx=np.zeros([rows,columns],np.uint8)
    for i in range(0,rows):
        for j in range(0,columns):
            if(X[i][j]>threshold):            
                outx[i][j]=255
    return outx

            

def createHistogram(img):
    rows,columns=img.shape
    d={}
    for i in range(0,rows):
        for j in range(0,columns):
            x=img[i][j]
            if(x in d):
                d[x]+=1
            else:
                d[x]=1
    x=list(d.keys())
    y=list(d.values())
    
    plt.bar(x[1:],y[1:],color='r')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('plot.jpg')
    plt.show()
    plt.close()
    
    l=[]
    l1=[]
    lrange=14
    for i in range(8,len(y)-8):
        new=y[i]
        newm=y[i-7]
        newplus=y[i+7]
        o=newplus-newm
        l.append(o)
        l1.append(i)
    new=max(l)
    for i in range(0,len(l)):
        if(l[i]==new):
            d=i
    f=l1[d]
    f+=lrange
    print('Threshold should be kept as: {}'.format(f))
    return f


def getCoordinates(img):
    rows,columns=img.shape
    l=[]
    for i in range(0,rows):
        for j in range(0,columns):
            
            if(img[i][j]==255):
                l.append([i,j])
    l=np.array(l)
    return l


def label(x,imgnew):
    for i in range(0,imgnew.shape[0]):
        for j in range(0,imgnew.shape[1]):
            for k in range(0,len(x)):
                if(i==x[k][0] and j==x[k][1]):
                    imgnew[i-1][j-1]=(255,0,0)
                    imgnew[i-1][j]=(255,0,0)
                    imgnew[i-1][j+1]=(255,0,0)
                    imgnew[i+1][j-1]=(255,0,0)
                    imgnew[i+1][j]=(255,0,0)
                    imgnew[i+1][j+1]=(255,0,0)
                    imgnew[i][j]=(255,0,0)
                    imgnew[i][j]=(255,0,0)
                    imgnew[i][j]=(255,0,0)
    cv2.imwrite('pointdisplay.jpg',imgnew)
    

#task 2.1
 

#g=createHistogram(img)
#print(g)
out=calc(img,300)
x=getCoordinates(out)
#print(x)
label(x,imgnew)
cv2.imshow('Point',out)
cv2.waitKey(0)

#coordinates [247, 443], [247, 444], [249, 443], [249, 444], [249, 445]




#task 2.2

threshold=createHistogram(img2)           
out2=segment(img2,203)
cv2.imwrite('segmented result.jpg',out2)
#cv2.imshow('Point2',out2)
#cv2.waitKey(0)

