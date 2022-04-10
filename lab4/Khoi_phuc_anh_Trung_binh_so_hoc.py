import numpy as np
import cv2
import matplotlib.pyplot as plt

def Loc_Trung_binh_so_hoc(img, ksize):
    m, n = img.shape
    img_ket_qua_anh_loc = np.zeros([m, n])
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            img_ket_qua_anh_loc[i,j] = np.mean(vung_anh_kich_thuoc_k)
    return img_ket_qua_anh_loc

if __name__ == "__main__":
    img_nhieu = cv2.imread('Anh_nhieu_de_loc_Trung_binh.tif', 0)
    ksize =5
    img_ket_qua_TBSH = Loc_Trung_binh_so_hoc(img_nhieu,ksize)

    fig = plt.figure(figsize=(16, 9))    
    (ax1, ax2) = fig.subplots(1, 2)        
    ax1.imshow(img_nhieu, cmap='gray')     
    ax1.set_title("ảnh gốc bị nhiễu Gaussian")           
    ax1.axis("off")

    ax2.imshow(img_ket_qua_TBSH, cmap='gray')      
    ax2.set_title("ảnh sau khi lọc Trung bình số học")
    ax2.axis("off")

    plt.show()