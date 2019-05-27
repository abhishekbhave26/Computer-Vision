# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 22:05:23 2018

@author: abhis
"""

import cv2
import numpy as np
import math

img=cv2.imread("C:/Users/abhis/Desktop/us/Courses/CVIP 573/homework and project/task2.jpg", 0)

height2 = img.shape[0]
width2 = img.shape[1]
e=0
dog_matrix=[]
count=0


#taking initial value of each octave
s=[0.5,math.sqrt(2),2*math.sqrt(2),4*math.sqrt(2)]
octave=[]
for i in range(0,4):
    row=[]
    newoct=s[i]*math.sqrt(2)
    for j in range(0,5):
        row.append(newoct)
        newoct*=math.sqrt(2)
    octave.append(row)

#initialization 
new =[]
for i in range(height2):        
    rowList = []
    for j in range(width2):                        
        rowList.append(0)
    new.append(rowList)  
        
#calculates each value of kernel      
def calcgaussiankernel(sigma):
    
    out=[]
    sum=0
    for i in range(0,7):
        row=[]
        for j in range(0,7):
            lower_part=(((j-3)**2 +(3-i)**2)/(2*sigma*sigma))
            a=float((1/(2*math.pi*sigma*sigma))*math.exp(-1*lower_part))
            sum+=a
            row.append(a)
        out.append(row)
    x=np.array(out,dtype=np.float)/sum
    return x          

#gaussian blur
def calcimg(blur,X,c,d):
    
    height=X.shape[0]
    width=X.shape[1]
    outx=[]
    for i in range(3,height-3):
        row=[]
        for j in range(3,width-3):
            
            s= (blur[0][0]*X[i-3][j-3]+blur[0][1]*X[i-3][j-2]+blur[0][2]*X[i-3][j-1]+blur[0][3]*X[i-3][j]+blur[0][4]*X[i-3][j+1]+blur[0][5]*X[i-3][j+2]+blur[0][6]*X[i-3][j+3]+\
                blur[1][0]*X[i-3][j-2]+blur[1][1]*X[i-2][j-2]+blur[1][2]*X[i-2][j-1]+blur[1][3]*X[i-2][j]+blur[1][4]*X[i-2][j+1]+blur[1][5]*X[i-2][j+2]+blur[1][6]*X[i-2][j+3]+\
                blur[2][0]*X[i-3][j-1]+blur[2][1]*X[i-1][j-2]+blur[2][2]*X[i-1][j-1]+blur[2][3]*X[i-1][j]+blur[2][4]*X[i-1][j+1]+blur[2][5]*X[i-1][j+2]+blur[2][6]*X[i-1][j+3]+\
                blur[3][0]*X[i-3][j]+blur[3][1]*X[i][j-2]+blur[3][2]*X[i][j-1]+blur[3][3]*X[i][j]+blur[3][4]*X[i][j+1]+blur[3][5]*X[i][j+2]+blur[3][6]*X[i][j+3]+\
                blur[4][0]*X[i-3][j+1]+blur[4][1]*X[i+1][j-2]+blur[4][2]*X[i+1][j-1]+blur[4][3]*X[i+1][j]+blur[4][4]*X[i+1][j+1]+blur[4][5]*X[i+1][j+2]+blur[4][6]*X[i+1][j+3]+\
                blur[5][0]*X[i-3][j+2]+blur[5][1]*X[i+2][j-2]+blur[5][2]*X[i+2][j-1]+blur[5][3]*X[i+2][j]+blur[5][4]*X[i+2][j+1]+blur[5][5]*X[i+2][j+2]+blur[5][6]*X[i+2][j+3]+\
                blur[6][0]*X[i-3][j+3]+blur[6][1]*X[i+3][j-2]+blur[6][2]*X[i+3][j-1]+blur[6][3]*X[i+3][j]+blur[6][4]*X[i+3][j+1]+blur[6][5]*X[i+3][j+2]+blur[6][6]*X[i+3][j+3])
            row.append(s)
        outx.append(row)
    a=np.array(outx,dtype=np.uint8)
    count=str(c)+str(d)
    cv2.imwrite('Blur'+str(count)+'.jpg',a)
    
#selects octave value
def calcsigma(octno,octlength):
    return octave[octno][octlength]


#generates dog
def dog(img2,img1,c,d):
    m=img2.shape[0]
    n=img2.shape[1]
    dog=[]
    for i in range(0,m):
        row=[]
        for j in range(0,n):
            x=int(img2[i][j])-int(img1[i][j])
            row.append(x)
        dog.append(row)
    dog1=np.array(dog,dtype=np.uint8)
    count=str(c)+str(d)
    cv2.imwrite('DoG'+str(count)+'.jpg',dog1)
    return dog1

#resizes image by 1/2 each time
def resize(img,m,n,a):
    resize=[]
    a+=1
    for i in range(0,m):
        row=[]
        for j in range(0,n):
            if(j%(2**a)==0 and i%(2**a)==0):
                row.append(img[i][j])
            else:
                pass
        resize.append(row)
        
    temp=resize[::2**a]
    newimg=np.array(temp,dtype=np.uint8)
    return newimg

#function to generate keypoints and 3 images            
def genkeypoints():
    for a in range(0,4):        
        for b in range(1,3):
            c = int(b-1)
            d = 'DoG' + str(a) + str(c) + '.jpg'
            prev = cv2.imread(d,0)
            e = 'DoG' + str(a) + str(b) + '.jpg'
            img = cv2.imread(e,0)
            f = int(b+1)
            g = 'DoG' + str(a) + str(f) + '.jpg'
            next1 = cv2.imread(g,0)
            keypoint(prev, img, next1, a) 
            
        #displaying final image of each octave
    octaveimg = np.array(blank, dtype=np.uint8)
    d = 'Octave.jpg'
    cv2.imshow(d,octaveimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
        
    cv2.imwrite(d,octaveimg)
    
def keypoint(prev, img, next1, a):
    
    m = img.shape[0]
    n = img.shape[1]
    
    for i in range(1,m-1):
        for j in range(1,n-1):            
            temp = img[i][j]
            #counter to check if all conditions are satisfied
            min = False
            max = False
            #comparing with its own 8 values
            if temp < img[i-1][j-1] and temp < img[i-1][j] and temp < img[i-1][j+1] and temp < img[i][j-1] and temp < img[i][j+1] and temp < img[i+1][j-1] and temp < img[i+1][j] and temp < img[i+1][j+1]:
                min = True
            if temp > img[i-1][j-1] and temp > img[i-1][j] and temp > img[i-1][j+1] and temp > img[i][j-1] and temp > img[i][j+1] and temp > img[i+1][j-1] and temp > img[i+1][j] and temp > img[i+1][j+1]:
                max = True
            
            #comparing with values of prev image
            if min == True or max == True:
                if min == True:
                    if temp < prev[i-1][j-1] and temp < prev[i-1][j] and temp < prev[i-1][j+1] and temp < prev[i][j-1] and temp < prev[i][j+1] and temp < prev[i+1][j-1] and temp < prev[i+1][j] and temp < prev[i+1][j+1]:
                        min= True
                    else:
                        min = False
                if max == True:        
                    if temp > prev[i-1][j-1] and temp > prev[i-1][j] and temp > prev[i-1][j+1] and temp > prev[i][j-1] and temp > prev[i][j+1] and temp > prev[i+1][j-1] and temp > prev[i+1][j] and temp > prev[i+1][j+1]:
                        max == True
                    else:
                        max = False            
            
            #comparing with values of next image
            if min == True or max == True:
                if min == True:
                    if temp < next1[i-1][j-1] and temp < next1[i-1][j] and temp < next1[i-1][j+1] and temp < next1[i][j-1] and temp < next1[i][j+1] and temp < next1[i+1][j-1] and temp < next1[i+1][j] and temp < next1[i+1][j+1]:
                        min= True
                    else:
                        min = False
                if max == True:        
                    if temp > next1[i-1][j-1] and temp > next1[i-1][j] and temp > next1[i-1][j+1] and temp > next1[i][j-1] and temp > next1[i][j+1] and temp > next1[i+1][j-1] and temp > next1[i+1][j] and temp > next1[i+1][j+1]:
                        max == True
                    else:
                        max = False
                        
            #plotting the point in the empty image
            if min == True or max == True:
                I = i*2**a
                J = j*2**a
                blank[I][J] = 255
  
# all fuctions call to create kernel,convolve it and resize
for x in range(0,4):
    for y in range(0,5):
        sigma=calcsigma(x,y)
        filter=calcgaussiankernel(sigma)
        calcimg(filter,img,x,y)
    
    img=cv2.imread("C:/Users/abhis/Desktop/us/Courses/CVIP 573/homework and project/task2.jpg", 0)
    height2 = img.shape[0]
    width2 = img.shape[1]
        
    img=resize(img,height2,width2,x)
    
#print('calc part done')

#generates a numpy array of dog
for x in range(0,4):
    dog_rows=[]
    for y in range(0,4):
        e=int(y+1)
        c='Blur'+str(x)+str(e)+'.jpg'
        img2=cv2.imread(c,0)
        
        d='Blur'+str(x)+str(y)+'.jpg'
        img1=cv2.imread(d,0)
        new=dog(img2,img1,x,y)        
        dog_rows.append(new)
    dog_matrix.append(dog_rows)
 
#print('Dog done')

blank =[]
for i in range(height2):        
    rowList = []
    for j in range(width2):                        
        rowList.append(0)
    blank.append(rowList) 

#function to calculate maxima and minima
genkeypoints()           


c= 0
list = []
for i in range(0,height2):
    for j in range(0,30):
        if blank[i][j] == 255:                
            c += 1
            list.append((i,j))
            break
            
    if c == 5:
            break
print(list)           
            
            

    

