import cv2
import numpy as np

def cursordetection():
    img=cv2.imread('C:/Users/abhis/Desktop/us/Courses/CVIP 573/homework and project/task3/set b/t2_4.jpg')
    img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    template = cv2.imread('temp1.png',0)
    template2 = cv2.imread('temp2.png',0)
    template3 = cv2.imread('temp3.png',0)
    
    w=template.shape[1]
    h = template.shape[0]
    w2=template2.shape[1]
    h2= template2.shape[0]
    w3=template3.shape[1]
    h3= template3.shape[0]
    
    blur_image=cv2.GaussianBlur(img_grey,(3,3),0)
    
    laplacian_blur_image = cv2.Laplacian(blur_image,cv2.CV_64F)
    laplacian_blur_template = cv2.Laplacian(template,cv2.CV_64F)
    laplacian_blur_template2 = cv2.Laplacian(template2,cv2.CV_64F)
    laplacian_blur_template3 = cv2.Laplacian(template3,cv2.CV_64F)
    
    new=np.asarray(laplacian_blur_image,dtype=np.float32)
    new1=np.asarray(laplacian_blur_template,dtype=np.float32)
    new2=np.asarray(laplacian_blur_template2,dtype=np.float32)
    new3=np.asarray(laplacian_blur_template3,dtype=np.float32)
    
    ssd = cv2.matchTemplate(new, new1,cv2.TM_CCOEFF_NORMED)
    ssd2 = cv2.matchTemplate(new,new2,cv2.TM_CCOEFF_NORMED)
    ssd3 = cv2.matchTemplate(new,new3,cv2.TM_CCOEFF_NORMED)
    threshold=0.45
    loc=np.where(ssd>=threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,0,255),2 )
    
    threshold2=0.33
    loc=np.where(ssd2>=threshold2)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img,pt,(pt[0]+w2,pt[1]+h2),(0,0,255),2 )
    
    threshold3=0.58
    loc=np.where(ssd3>=threshold3)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img,pt,(pt[0]+w3,pt[1]+h3),(0,0,255),2 )
    
    cv2.imshow('new.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cursordetection()  