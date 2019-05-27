# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 23:06:16 2018

@author: abhis
"""

import cv2
import numpy as np
import random
UBIT = 'abhave'
np.random.seed(sum([ord(c) for c in UBIT]))

#1st part
def doSIFT1():
    #1st part
    img1 = cv2.imread('tsucuba_left.png')
    gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    org1=gray1
    sift = cv2.xfeatures2d.SIFT_create()
    kp1,des1= sift.detectAndCompute(gray1,None)
    gray1=cv2.drawKeypoints(gray1,kp1,gray1)
    cv2.imwrite('task2_sift1.jpg',gray1)
    
    return gray1,kp1,org1,des1

def doSIFT2():
    
    img2 = cv2.imread('tsucuba_right.png')
    gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    org2=gray2
    sift = cv2.xfeatures2d.SIFT_create()
    kp2,des2 = sift.detectAndCompute(gray2,None)
    gray2=cv2.drawKeypoints(gray2,kp2,gray2)
    cv2.imwrite('task2_sift2.jpg',gray2)
    
    return gray2,kp2,org2,des2


def knnAndepiline(img1,img2,kp1,kp2,org1,org2,des1,des2):

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    new= []
    new2=[]
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            new.append([m])
            new2.append(m)
    
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,new,img1,flags=2)
    cv2.imwrite('task2_matches_knn.jpg',img3)
    F=fundamental(new2,img1,img2,kp1,kp2)
    return img3,F

    
#2nd part     
def fundamental(new,img1,img2,kp1,kp2):
    
    org1 = cv2.imread('tsucuba_left.png')
    org2 = cv2.imread('tsucuba_right.png')
    
    src_pts = np.int32([ kp1[m.queryIdx].pt for m in new ])
    dst_pts = np.int32([ kp2[m.trainIdx].pt for m in new ])
    #print(src_pts.shape,dst_pts.shape)
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC)
    #print(F)
    
    src_pts = src_pts[mask.ravel()==1]
    dst_pts= dst_pts[mask.ravel()==1]
    orgsrc=src_pts
    orgdst=dst_pts
    
    length=len(src_pts)
    list=[(255,100,0),(0,0,3),(255,0,0),(255,255,0),(0,0,255),(255,0,255),(255,255,255),(0,255,255),(100,255,0),(255,205,100)]    
    #print(len(list))
    for i in range(0,len(list)):
        
        x=random.randint(0,length-1)
        y=x+1
        src_pts=orgsrc[x:y]
        dst_pts=orgdst[x:y]
        #print(dst_pts)
        
        # right image and drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(dst_pts, 2,F)
        lines1 = lines1.reshape(-1,3)
        img5,img6 = drawlines(org1,org2,lines1,src_pts,dst_pts,color=list[i])
        # left image and drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(src_pts, 1,F)
        lines2 = lines2.reshape(-1,3)
        img3,img4 = drawlines(org2,org1,lines2,dst_pts,src_pts,color=list[i])
    
    cv2.imwrite('task2_epi_right.jpg',img3)
    cv2.imwrite('task2_epi_left.jpg',img5)

    return F  


def drawlines(img1,img2,lines,src_pts,dst_pts,color):
    
    r,c = img1.shape[:2]
    for r,pt1,pt2 in zip(lines,src_pts,dst_pts):
        #color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
       
    return img1,img2

#4th part
def disparityMap():
    img1 = cv2.imread('tsucuba_left.png')
    img2 = cv2.imread('tsucuba_right.png')
    
    window_size=5
    stereo = cv2.StereoSGBM_create(numDisparities=32, blockSize=15,speckleWindowSize=10,
                                   speckleRange=1,uniquenessRatio=3,preFilterCap=2,disp12MaxDiff=2,
                                   minDisparity=0,P1=8*3*window_size**2,P2=32*3*window_size**2)
    disparity = stereo.compute(img1,img2)
    cv2.imwrite('task2_disparity.jpg',disparity)
    
        
def main():    
    img1,kp1,org1,des1=doSIFT1()
    img2,kp2,org2,des2=doSIFT2()
    img3,F=knnAndepiline(img1,img2,kp1,kp2,org1,org2,des1,des2)
    print('The fundamental matrix is as follows : ')
    print(F)
    disparityMap()

main()



