import cv2     
import numpy as np   
import matplotlib.pyplot as plt 

def Filter(img,mask):
    m, n = img.shape
    img_new = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp   =  img[i-1, j-1]    * mask[0, 0]\
                   +  img[i-1, j]      * mask[0, 1]\
                   +  img[i-1, j + 1]  * mask[0, 2]\
                   +  img[i, j-1]      * mask[1, 0]\
                   +  img[i, j]        * mask[1, 1]\
                   +  img[i, j + 1]    * mask[1, 2]\
                   +  img[i + 1, j-1]  * mask[2, 0]\
                   +  img[i + 1, j]    * mask[2, 1]\
                   +  img[i + 1, j + 1]* mask[2, 2]
            img_new[i, j]= temp
    img_new = img_new.astype(np.uint8)
    return img_new

Gaussian = np.array(([0.0751/4.8976, 0.1238/4.8976, 0.0751/4.8976],
                           [0.1238/4.8976, 0.2042/4.8976, 0.1238/4.8976],
                           [0.0751/4.8976, 0.1238/4.8976, 0.0751/4.8976]), dtype="float")
locLaplace = np.array(([0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]), dtype="float")

fig = plt.figure(figsize=(16, 9)) 
ax1, ax2 = fig.subplots(1, 2)

image = cv2.imread('gaussian.tif', 0)
ax1.imshow(image, cmap='gray')
ax1.set_title("Ảnh gốc")

imgGaussian = Filter(image, Gaussian) 

img_loc_Laplace = Filter(imgGaussian, locLaplace) 

img_cai_thien_locLoG = imgGaussian - img_loc_Laplace
ax2.imshow(img_cai_thien_locLoG, cmap='gray')
ax2.set_title("Ảnh sau khi lọc LoG")



plt.show()