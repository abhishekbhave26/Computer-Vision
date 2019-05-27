# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:10:41 2018

@author: abhis
"""


import cv2
import numpy as np
import math
import operator
 


#convolution function
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
    #edges = cv2.Canny(img, 50, 250)
    #cv2.imshow('Edges',edges)
    #cv2.waitKey(0)
    
    
    #flipped sobel operator
    Gx=np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]],dtype=np.float)
    
    #flipped sobel operator
    Gy=np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]],dtype=np.float)
    
    G,G1=convolve(img,Gx,Gy)      
    
    cv2.imshow('Diagonla',G)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    #cv2.imshow('verti',G1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 
    return G


def hough(img):
    r,c=img.shape
    theta=np.linspace(-90.00,0,91)
    tres=1
    rhores=1
    theta = np.concatenate((theta, -theta[len(theta)-2::-1]))
    
    diag=np.sqrt((r*r)+(c*c))
    new=np.ceil(diag/rhores)
    nOfrho=2*new+1
    rho=np.linspace(-new*rhores,new*rhores,nOfrho)
    
    
    H=np.zeros([len(rho),len(theta)])
    for i in range(r):
        for j in range(c):
            x=img[i][j]
            if(x!=0):
                for k in range(len(theta)):
                    rval=j*np.cos(theta[k]*np.pi/180.0)+i*np.sin(theta[k]*np.pi/180.0)
                    for l in range(len(rho)):
                        if(rho[l]>rval):
                            break
                    H[l][k] += 1
            
            else:
                pass
    
    return rho,theta,H



def voting(H,rho,theta):
    a=len(rho)
    b=len(theta)
    number_of_lines=20
    d={}
    for i in range(0,a):
        l=[]
        for j in range(0,b):
           l.append(H[i][j])
           #Min=min(l)
           #Max=max(l)
           x=H[i][j]
           key=(i,j)
           if(key in d):
               d[key]+=x
           else:
               d[key]=x
    dcopy=d
    l=[]
    i=0
    while(i<number_of_lines):
        a=max(dcopy.items(), key=operator.itemgetter(1))[0]
        l.append(a)
        #print(l)
        del dcopy[a]
        i+=1
    
    #print(l)
    return l


def hough_lines_draw(img, indicies, rhos, thetas):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. '''
    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        
        
        
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def valid_point(pt, ymax, xmax):
  '''
  @return True/False if pt is with bounds for an xmax by ymax image
  '''
  x, y = pt
  if x <= xmax and x >= 0 and y <= ymax and y >= 0:
    return True
  else:
    return False

def round_tup(tup):
  '''
  @return closest integer for each number in a point for referencing
  a particular pixel in an image
  '''
  x,y = [int(round(num)) for num in tup]
  return (x,y)



def draw_rho_theta_pairs(target_im, pairs):
  '''
  @param opencv image
  @param array of rho and theta pairs
  Has the side-effect of drawing a line corresponding to a rho theta
  pair on the image provided
  '''
  im_y_max, im_x_max= np.shape(target_im)
  for i in range(0, len(pairs), 1):
    point = pairs[i]
    rho = point[0]
    theta = point[1] * np.pi / 180 # degrees to radians
    # y = mx + b form
    m = -np.cos(theta) / np.sin(theta)
    b = rho / np.sin(theta)
    # possible intersections on image edges
    left = (0, b)
    right = (im_x_max, im_x_max * m + b)
    top = (-b / m, 0)
    bottom = ((im_y_max - b) / m, im_y_max)

    pts = [pt for pt in [left, right, top, bottom] if valid_point(pt, im_y_max, im_x_max)]
    if len(pts) == 2:
      cv2.line(target_im, round_tup(pts[0]), round_tup(pts[1]), (0,0,255), 1)
  cv2.imshow('result', target_im)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
 
      
      


img=cv2.imread('hough.jpg',0)
edges=sobel()
rho,theta,H=hough(edges)
rt=voting(H,rho,theta)
hough_lines_draw(img,rt,rho,theta)
draw_rho_theta_pairs(img,rt)




