# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 23:34:55 2018

@author: abhis
"""

import cv2
import numpy as np
import operator


img = cv2.imread('hough.jpg')
imgn = img[:,:,::-1]
lines2 = imgn.copy()
lines = imgn.copy()


def convolve(X,Gx,Gy):
    rows=X.shape[0]
    columns=X.shape[1]
    
    outx=np.zeros([rows,columns])
    outy=np.zeros([rows,columns])
    
    for i in range(1,rows-1):
        for j in range(1,columns-1):
            
            s=Gx[0][0]*X[i-1][j-1]+Gx[0][1]*X[i-1][j]+Gx[0][2]*X[i-1][j+1]+\
                Gx[1][0]*X[i][j-1]+Gx[1][1]*X[i][j]+Gx[1][2]*X[i][j+1]+\
                Gx[2][0]*X[i+1][j-1]+Gx[2][1]*X[i+1][j]+Gx[2][2]*X[i+1][j+1]
            
            p=Gy[0][0]*X[i-1][j-1]+Gy[0][1]*X[i-1][j]+Gy[0][2]*X[i-1][j+1]+\
                Gy[1][0]*X[i][j-1]+Gy[1][1]*X[i][j]+Gy[1][2]*X[i][j+1]+\
                Gy[2][0]*X[i+1][j-1]+Gy[2][1]*X[i+1][j]+Gy[2][2]*X[i+1][j+1]
            
            if(s>30 or p>30):
                
                outx[i-1][j-1]=s
                outy[i-1][j-1]=p
    
    return outx,outy



def sobel():
    img = cv2.imread("hough.jpg",0)
    
    #flipped sobel operator
    Gx=np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]],dtype=np.float)

    #flipped sobel operator
    Gy=np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]],dtype=np.float)
    
    G,G1=convolve(img,Gx,Gy)      

    return G,G1



def hough(img,x ,y ,ss):
    r,c=img.shape
    theta=np.linspace(x,y,30.00 + ss)
    tres=1
    rhores=1
    theta = np.concatenate((theta, -theta[len(theta)-2::-1]))
    
    diag=np.sqrt((r*r)+(c*c))
    new=np.ceil(diag/rhores)
    nOfrho=2*new+1
    rho=np.linspace(-new,new*rhores,nOfrho)
    
    
    H=np.zeros([len(rho),len(theta)])
    for i in range(r):
        for j in range(c):
            x=img[i][j]
            if(x!=0):
                for k in range(len(theta)):
                    p=j*np.cos(theta[k]*np.pi/180.0)
                    q=i*np.sin(theta[k]*np.pi/180.0)
                    rval=p+q
                    for l in range(len(rho)):
                        if(rho[l]>rval):
                            break
                    H[l][k] += 1
            
            else:
                pass
    
    return rho,theta,H


def voting_peaks(H,rho,theta):
    a=len(rho)
    b=len(theta)
    number_of_lines=30
    d={}
    l=[]
    lnew=[]
    for i in range(0,b):
        for j in range(0,a):
           x=H[j][i]
           key=(j,i)
           
           if(key in d):
               d[key]+=x
           else:
               d[key]=x
    dcopy=d
    count=0
    while(count<number_of_lines):
        a=max(dcopy.items(), key=operator.itemgetter(1))[0]
        new=a[::-1]
        l.append(new)
        lnew.append(a)
        #print(l)
        del dcopy[a]
        count+=1
    
    rho_theta=[]
    for i in range(0,len(lnew)):
        val=lnew[i]
        x = rho[val[0]]
        y = theta[[val[1]]]
        rho_theta.append([x,y[0]])
    
    return l,rho_theta



def vpoint(pt, ymax, xmax):
    
    flag=False
    flag2=False
    x1, y1 = pt
    if(x1 <= xmax and x1 >= 0):
        flag=True
    if(y1 <= ymax and y1 >= 0):
        flag2=True
    if(flag==True and flag2==True):
        return True
    else:
        return False



def roundingInt(t):
    i,j = [int(round(num)) for num in t]
    return (i,j)



def draw(imgnew,pairs,p,q,r,save):
    
    ym, xm, c = np.shape(imgnew)
    for i in range(0, len(pairs), 1):
        
        point = pairs[i]
        rho = point[0]
        theta = point[1] * np.pi / 180 
        m = -np.cos(theta) / np.sin(theta)
        b = rho / np.sin(theta)
        left_corner = (0, b)
        right_corner = (xm, xm * m + b)
        top_corner = (-b / m, 0)
        bottom_corner = ((ym - b) / m, ym)

        pts = [pt for pt in [left_corner, right_corner, top_corner, bottom_corner] if vpoint(pt, ym, xm)]
        if len(pts) == 2:            
            cv2.line(imgnew, roundingInt(pts[0]), roundingInt(pts[1]), (p,q,r), 2)            
    cv2.imwrite(save,imgnew)
  



edges,ed=sobel()
print('For Vertical Lines')
rhos, thetas, H = hough(edges,0.0,35.0,1.0)
xy,rt=voting_peaks(H,rhos,thetas)
draw(lines, rt,0,0,255,'red_line.jpg')



print('For Diagonal Lines')
rhos2, thetas2, H2 = hough(ed,-60,-30,0.0)
xy2,rt2=voting_peaks(H2,rhos2,thetas2)
draw(lines2, rt2,255,0,0,'blue_lines.jpg')

'''
print('For Circle')
rhos2, thetas2, H2 = hough(edges,0,360,0.0)
xy2,rt2=voting_peaks(H2,rhos2,thetas2)
draw(lines2, rt2,255,0,0,'circle.jpg')
'''