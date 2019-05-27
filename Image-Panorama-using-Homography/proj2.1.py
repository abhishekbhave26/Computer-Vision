# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:13:25 2018

@author: abhis
"""

import cv2
import numpy as np
import random
UBIT = 'abhave'
np.random.seed(sum([ord(c) for c in UBIT]))

#1st part
def doSIFT1():

    img1 = cv2.imread('mountain1.jpg')
    gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    org1=gray1
    sift = cv2.xfeatures2d.SIFT_create()
    kp1,des1= sift.detectAndCompute(gray1,None)
    gray1=cv2.drawKeypoints(gray1,kp1,gray1)
    cv2.imwrite('task1_sift1.jpg',gray1)
    
    return gray1,kp1,org1,des1

def doSIFT2():
    
    img2 = cv2.imread('mountain2.jpg')
    gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    org2=gray2
    sift = cv2.xfeatures2d.SIFT_create()
    kp2,des2 = sift.detectAndCompute(gray2,None)
    gray2=cv2.drawKeypoints(gray2,kp2,gray2)
    cv2.imwrite('task1_sift2.jpg',gray2)
    
    return gray2,kp2,org2,des2
    

#2nd part   
def knn(img1,img2,kp1,kp2,org1,org2,des1,des2):
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    #print(matches)
    new = []
    new1=[]
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            new.append([m])
            new1.append(m)
            
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,new,img1,flags=2)
    cv2.imwrite('task1_matches_knn.jpg',img3)
    
    M=homographyDrawMatchesAndWarp(new1,img1,img2,kp1,kp2)
    
    return img3,M
    


#3rd ,4th and 5th part
def homographyDrawMatchesAndWarp(new,img1,img2,kp1,kp2):
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in new]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in new]).reshape(-1,1,2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    length=len(matchesMask)
    x=random.randint(0,length-10)
    y=x+10
    draw_params = dict(matchColor = (0,255,0), singlePointColor = None,matchesMask = matchesMask[x:y],flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,new[x:y],None,**draw_params)
    cv2.imwrite('task1_matches.jpg',img3)
    
    
    img1 = cv2.imread('mountain1.jpg')
    img2 = cv2.imread('mountain2.jpg')
    
    r1, c1 = img1.shape[:2]
    r2, c2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,r1], [c1, r1], [c1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,r2], [c2, r2], [c2,0]]).reshape(-1,1,2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, M)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [xmin, ymin] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    output_img = cv2.warpPerspective(img1, H_translation.dot(M), (xmax - xmin, ymax - ymin))
    output_img[translation_dist[1]:r1+translation_dist[1],translation_dist[0]:c1+translation_dist[0]] = img2

    cv2.imwrite('task1_pano.jpg',output_img)
    return M


def main():    
    img1,kp1,org1,des1=doSIFT1()
    img2,kp2,org2,des2=doSIFT2()
    img3,M=knn(img1,img2,kp1,kp2,org1,org2,des1,des2)  
    print('The homography matrix is as follows : ')
    print(M)
    #[[ 1.58720376e+00 -2.91747553e-01 -3.95226519e+02]
    #[ 4.48097764e-01  1.43063310e+00 -1.90273584e+02]
    #[ 1.20808480e-03 -6.07787702e-05  1.00000000e+00]]

main()
