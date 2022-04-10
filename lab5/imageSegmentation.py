import sys
import cv2
import time
import numpy as np




def segment(A, Channel, K, Enhance):
   
    if Enhance != 1 and Enhance != 0:
        print ('Invalid input for Enhance Parameter, must be 1 or 0')
        return
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)

   
    if len(Channel) == 1:
        Z = A[:, :, Channel].copy()
        Z = Z.reshape((-1, 1))
        Z = np.float32(Z)

        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((A[:, :, Channel].shape))
        A[:, :, Channel] = res2

  
    elif len(Channel) == 2:
        Channel1 = Channel[0]
        Channel2 = Channel[1]

        Z1 = A[:, :, Channel1].copy()
        Z2 = A[:, :, Channel2].copy()
        Z1 = Z1.reshape((-1, 1))
        Z2 = Z2.reshape((-1, 1))
        Z = np.hstack((Z1, Z2))
        Z = np.float32(Z)
 
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((A.shape[0], A.shape[1], 2))
 
        if Enhance == 0:
            A[:, :, Channel1] = res2[:, :, 0]
        elif Enhance == 1:
            A[:, :, Channel1] = res2[:, :, 1]
        A[:, :, Channel2] = res2[:, :, 1]

   
    elif len(Channel) == 3:
        Z = A.copy()
        Z = Z.reshape((-1, 3))
        Z = np.float32(Z)
       
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
       
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((A.shape))
        A[:, :, :] = res2[:, :, :]



img = cv2.imread('flowers.png', 1)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


segment(img_hsv, (0, 2), 3, 0)

img1 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('HSV segmentation', img1)


segment(img, (1, 2), 4, 1)
cv2.imshow('RGB segmentation', img)



cv2.waitKey(0)
