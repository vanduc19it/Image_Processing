import cv2    
import numpy as np   
import matplotlib.pyplot as plt 

def Filter(img,mask):
    m, n = img.shape
    img_new = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp   =  img[i-1, j-1]    * mask[0, 0]\
                   +  img[i, j-1]      * mask[0, 1]\
                   +  img[i+1, j - 1]  * mask[0, 2]\
                   +  img[i-1, j]      * mask[1, 0]\
                   +  img[i, j]        * mask[1, 1]\
                   +  img[i+1, j]      * mask[1, 2]\
                   +  img[i - 1, j+1]  * mask[2, 0]\
                   +  img[i, j + 1]    * mask[2, 1]\
                   +  img[i + 1, j + 1]* mask[2, 2]
            img_new[i, j]= temp
    img_new = img_new.astype(np.uint8)
    return img_new

locLaplace = np.array(([0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]), dtype="float")

fig = plt.figure(figsize=(16, 9)) 
ax1, ax2 = fig.subplots(1, 2) 

image = cv2.imread('laplace.tif', 0)
ax1.imshow(image, cmap='gray')
ax1.set_title("Ảnh gốc")

img_loc_Laplace = Filter(image, locLaplace) 

img_cai_thien_locLaplace = image - img_loc_Laplace
ax2.imshow(img_cai_thien_locLaplace, cmap='gray')
ax2.set_title("Ảnh sau khi lọc Laplace")

plt.show()
