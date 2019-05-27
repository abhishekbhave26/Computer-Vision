# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 23:45:26 2018

@author: abhis
"""

import numpy as np
import math
from matplotlib import pyplot as plt
import cv2

UBIT = 'abhave'
np.random.seed(sum([ord(c) for c in UBIT]))

list=[[5.9, 3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]]
#x=np.array(list)
k=3
n=[[6.2,3.2],[6.6,3.7],[6.5,3.0]] 
    #red       #green    #blue
list=np.array(list)
n=np.array(n)    


def iteration(list,n):
    list1=[]
    list2=[]
    list3=[]
    for i in range(0,len(list)):
        x1=list[i][0]
        y1=list[i][1]
        row=[]
        for j in range(0,3):
            x2=n[j][0]
            y2=n[j][1]
            dist=math.sqrt((x2-x1)**2+(y2-y1)**2)
            row.append(dist)
        x=np.min(row)
        if(row[0]==x):
            list1.append([x1,y1])
        elif(row[1]==x):
            list2.append([x1,y1])
        else:
            list3.append([x1,y1])
    
    l1=np.asarray(list1,dtype='float64')
    l2=np.asarray(list2,dtype='float64')
    l3=np.asarray(list3,dtype='float64')
    return l1,l2,l3
    
def reomputeCentroid(list):
    x=len(list)
    #print(x)
    new=[]
    sumx=0
    sumy=0
    for i in range(0,x):
        sumx=sumx+list[i][0]
        sumy=sumy+list[i][1]
        #print(sumx,sumy)
    sumx,sumy=sumx/x,sumy/x
    new.append(sumx)
    new.append(sumy)
    return new


def colorPoints1(list1,list2,list3,n,str):
    
    plt.scatter(list1[:,0],list1[:,1], edgecolors='red', facecolor='red',marker="^")
    plt.scatter(list2[:,0],list2[:,1],edgecolors='green', facecolor='green',marker="^")
    plt.scatter(list3[:,0],list3[:,1],edgecolors='blue', facecolor='blue',marker="^")
    
    plt.scatter(n[0][0],n[0][1],edgecolors='red', facecolor='red',marker='+')
    plt.scatter(n[1][0],n[1][1],edgecolors='green', facecolor='green',marker='+')
    plt.scatter(n[2][0],n[2][1],edgecolors='blue', facecolor='blue',marker='+')
    plt.savefig(str)
    plt.show()
    
    plt.close()


def baboon(k):
    
    img=cv2.imread('baboon.jpg')
    n=[]
    for i in range(0,k):
        n.append(img[0,i])
    n=np.array(n)
    l1,l2,l3=baboonIteration(img,n)
    #print(img[0][2][0])    
    #print(n)
    new=[]
    new.append(l1)
    new.append(l2)
    new.append(l3)
    return new
    
    
def baboonIteration(img,n):
    list1=[]
    list2=[]
    list3=[]
    for i in range(0,len(img)):
        for x in range(0,len(img)):
            
            x1=img[i][x][0]
            y1=img[i][x][1]
            z1=img[i][x][2]
            row=[]
            for j in range(0,3):
                x2=n[j][0]
                y2=n[j][1]
                z2=n[j][2]
                dist=math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
                row.append(dist)
            x=np.min(row)
            if(row[0]==x):
                list1.append([x1,y1,z1])
            elif(row[1]==x):
                list2.append([x1,y1,z1])
            else:
                list3.append([x1,y1,z1])
    
    l1=np.asarray(list1,dtype='float64')
    l2=np.asarray(list2,dtype='float64')
    l3=np.asarray(list3,dtype='float64')
    return l1,l2,l3



def main():
    str='task3_iter1_a.png'    
    list1,list2,list3=iteration(list,n)
    colorPoints1(list1,list2,list3,n,str)
    print('The clusters are as follows :')
    print(list1)
    print(list2)
    print(list3)
    
    
    n[0]=reomputeCentroid(list1)
    n[1]=reomputeCentroid(list2)
    n[2]=reomputeCentroid(list3)
    print('The updateded centroids are as follows :')
    print(n)
    
    str='task3_iter1_b.png'
    colorPoints1(list1,list2,list3,n,str)
    
    
    str='task3_iter2_a.png'
    list1,list2,list3=iteration(list,n)
    colorPoints1(list1,list2,list3,n,str)
    print('The clusters are as follows :')
    print(list1)
    print(list2)
    print(list3)
    
    str='task3_iter2_b.png'
    n[0]=reomputeCentroid(list1)
    n[1]=reomputeCentroid(list2)
    n[2]=reomputeCentroid(list3)
    print('The updateded centroids are as follows :')
    print(n)
    colorPoints1(list1,list2,list3,n,str)


main()
#img=baboon(3)
#img=np.array(img).reshape(512,512,3)
#cv2.imshow('new.jpg',img)
