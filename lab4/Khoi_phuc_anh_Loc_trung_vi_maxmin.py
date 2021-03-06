import numpy as np
import cv2
import matplotlib.pyplot as plt


def Loc_TKTT_max(img, ksize):
    m, n = img.shape
    img_ket_qua_anh_loc_max = np.zeros([m, n])
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            gia_tri_Max = np.max(vung_anh_kich_thuoc_k)
            img_ket_qua_anh_loc_max[i, j] = gia_tri_Max
    return img_ket_qua_anh_loc_max

def Loc_TKTT_min(img, ksize):
    m, n = img.shape
    img_ket_qua_anh_loc_min = np.zeros([m, n])
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            gia_tri_Min = np.min(vung_anh_kich_thuoc_k)
            img_ket_qua_anh_loc_min[i, j] = gia_tri_Min
    return img_ket_qua_anh_loc_min

def Loc_TKTT_trung_vi(img, ksize):
    m, n = img.shape
    img_ket_qua_anh_loc_Trung_vi= np.zeros([m, n])
    h=(ksize -1) // 2
    padded_img = np.pad(img, (h, h), mode='reflect')
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize,j:j+ksize]
            gia_tri_TV = np.median(vung_anh_kich_thuoc_k)
            img_ket_qua_anh_loc_Trung_vi[i, j] = gia_tri_TV
    return img_ket_qua_anh_loc_Trung_vi

if __name__ == "__main__":

    img_nhieu_hat_tieu = cv2.imread('Anh_nhieu_hat_tieu.tif', 0)
    img_nhieu_muoi = cv2.imread('Anh_nhieu_muoi.tif', 0)
    img_nhieu_muoi_tieu = cv2.imread('Anh_nhieu_de_loc_Trung_vi.tif', 0)
    ksize_1 = 3 
    ksize_2 =7  
    img_KQ_Max = Loc_TKTT_max(img_nhieu_hat_tieu, ksize_1)
    img_KQ_Min = Loc_TKTT_min(img_nhieu_muoi, ksize_1)
    img_KQ_Trung_vi= Loc_TKTT_trung_vi(img_nhieu_muoi_tieu , ksize_2)

    fig = plt.figure(figsize=(16, 9))    
    (ax1, ax2), (ax3,ax4),(ax5,ax6) = fig.subplots(3, 2)   
    ax1.imshow(img_nhieu_hat_tieu, cmap='gray')      
    ax1.set_title("ảnh gốc bị nhiễu hạt tiêu")             
    ax1.axis("off")

    ax2.imshow(img_KQ_Max, cmap='gray')       
    ax2.set_title("ảnh sau khi lọc TKTT Max") 
    ax2.axis("off")

    ax3.imshow(img_nhieu_muoi, cmap='gray')  
    ax3.set_title("ảnh gốc bị nhiễu muối") 
    ax3.axis("off")

    ax4.imshow(img_KQ_Min, cmap='gray')  
    ax4.set_title("ảnh sau khi lọc TKTT Min") 
    ax4.axis("off")

    ax5.imshow(img_nhieu_muoi_tieu, cmap='gray')  
    ax5.set_title("ảnh gốc bị nhiễu muối tiêu") 
    ax5.axis("off")

    ax6.imshow(img_KQ_Trung_vi, cmap='gray')  
    ax6.set_title("ảnh sau khi lọc Trung vị")  
    ax6.axis("off")
    plt.show()