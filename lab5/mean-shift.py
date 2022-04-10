

import numpy as np
from scipy import ndimage
import math
from random import randint
import matplotlib.pyplot as plt
import cv2


Mode = 2

imgPath = 'Butterfly.jpg'

H = 90
Hr = 90
Hs = 90
Iter = 100


img = cv2.imread(imgPath,cv2.IMREAD_COLOR)
opImg = np.zeros(img.shape,np.uint8)
boundaryImg = np.zeros(img.shape,np.uint8)



def getNeighbors(seed,matrix,mode=1):
    neighbors = []
    nAppend = neighbors.append
    sqrt = math.sqrt
    for i in range(0,len(matrix)):
        cPixel = matrix[i]
        
        if (mode == 1):
            d = sqrt(sum((cPixel-seed)**2))
            if(d<H):
                 nAppend(i)
        
        else:
            r = sqrt(sum((cPixel[:3]-seed[:3])**2))
            s = sqrt(sum((cPixel[3:5]-seed[3:5])**2))
            if(s < Hs and r < Hr ):
                nAppend(i)
    return neighbors


def markPixels(neighbors,mean,matrix,cluster):
    for i in neighbors:
        cPixel = matrix[i]
        x=cPixel[3]
        y=cPixel[4]
        opImg[x][y] = np.array(mean[:3],np.uint8)
        boundaryImg[x][y] = cluster
    return np.delete(matrix,neighbors,axis=0)


def calculateMean(neighbors,matrix):
    neighbors = matrix[neighbors]
    r=neighbors[:,:1]
    g=neighbors[:,1:2]
    b=neighbors[:,2:3]
    x=neighbors[:,3:4]
    y=neighbors[:,4:5]
    mean = np.array([np.mean(r),np.mean(g),np.mean(b),np.mean(x),np.mean(y)])
    return mean


def createFeatureMatrix(img):
    h,w,d = img.shape
    F = []
    FAppend = F.append
    for row in range(0,h):
        for col in range(0,w):
            r,g,b = img[row][col]
            FAppend([r,g,b,row,col])
    F = np.array(F)
    return F


def performMeanShift(img):
    clusters = 0
    F = createFeatureMatrix(img)

    while(len(F) > 0):
        print ('remPixelsCount : ' + str(len(F)))

        randomIndex = randint(0,len(F)-1)
        seed = F[randomIndex]

        initialMean = seed

        neighbors = getNeighbors(seed,F,Mode)
        print('found neighbors :: '+str(len(neighbors)))

        if(len(neighbors) == 1):
            F=markPixels([randomIndex],initialMean,F,clusters)
            clusters+=1
            continue

        mean = calculateMean(neighbors,F)
 
        meanShift = abs(mean-initialMean)

        if(np.mean(meanShift)<Iter):
            F = markPixels(neighbors,mean,F,clusters)
            clusters+=1
    return clusters


def main():
    clusters = performMeanShift(img)
    origlabelledImage, orignumobjects = ndimage.label(opImg)

    cv2.imshow('Origial Image',img)
    cv2.imshow('OP Image',opImg)
    cv2.imshow('Boundry Image',boundaryImg)
    
    cv2.imwrite('temp.jpg',opImg)
    temp = cv2.imread('temp.jpg',cv2.IMREAD_COLOR)
    labels, numobjects = ndimage.label(temp)
    fig, ax = plt.subplots()
    ax.imshow(labels)
    ax.set_title('Labeled objects')
    
    print ('Number of clusters formed : ', clusters)


if __name__ == "__main__":
    main()