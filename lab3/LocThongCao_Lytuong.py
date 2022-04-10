import numpy as np
import matplotlib.pyplot as plt
import cv2

def DFT1D(img):
    U = len(img)
    outarry = np.zeros(U, dtype=complex)
    for m in range(U):
        sum = 0.0
        for n in range(U):
            e = np.exp(-1j * 2 * np.pi * m * n / U)
            sum += img[n] * e
        outarry[m] = sum
    return outarry

def IDFT1D(img):
    U = len(img)
    outarry = np.zeros(U,dtype=complex)
    for n in range(U):
        sum = 0.0
        for m in range(U):
            e = np.exp(1j * 2 * np.pi * m * n / U)
            sum += img[m]*e
        pixel = sum/U
        outarry[n]=pixel
    return outarry


def HighPass_Ideals(D0,U,V):
    H = np.zeros((U, V))
    D = np.zeros((U, V))
    U0 = int(U / 2)
    V0 = int(V / 2)
    
    for u in range(U):
        for v in range(V):
            u2 = np.power(u, 2)
            v2 = np.power(v, 2)
            D[u, v] = np.sqrt(u2 + v2)
    for u in range(U):
        for v in range(V):
            if D[np.abs(u - U0), np.abs(v - V0)] <= D0:
                H[u, v] = 0
            else:
                H[u, v] = 1
    return H

if __name__ == "__main__":
    
    image = cv2.imread("test.tif", 0)
    image = cv2.resize(src=image, dsize=(100, 100))
   
    f = np.asarray(image)
    M, N = np.shape(f)  

 
    P, Q = 2*M , 2*N
    shape = np.shape(f)
  
    f_xy_p = np.zeros((P, Q))
    f_xy_p[:shape[0], :shape[1]] = f


    F_xy_p = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            F_xy_p[x, y] = f_xy_p[x, y] * np.power(-1, x + y)

  
    dft_cot = dft_hang = np.zeros((P, Q))

    for i in range(P):
        dft_cot[i] = DFT1D(F_xy_p[i])
    
    for j in range(Q):
        dft_hang[:, j] = DFT1D(dft_cot[:, j])

 
    H_uv = HighPass_Ideals(10,P,Q)


    G_uv = np.multiply(dft_hang, H_uv)


    idft_cot = idft_hang = np.zeros((P, Q))
 
    for i in range(P):
        idft_cot[i] = IDFT1D(G_uv[i])

    for j in range(Q):
        idft_hang[:, j] = IDFT1D(idft_cot[:, j])


    g_array = np.asarray(idft_hang.real)
    P, Q = np.shape(g_array)
    g_xy_p = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            g_xy_p[x, y] = g_array[x, y] * np.power(-1, x + y)

    g_xy = g_xy_p[:shape[0], :shape[1]]


    fig = plt.figure(figsize=(16, 9)) 

    (ax1, ax2, ax3), (ax4, ax5, ax6),(ax7, ax8, ax9) = fig.subplots(3, 3)


    ax1.imshow(image, cmap='gray')
    ax1.set_title('Ảnh gốc MxN')
    ax1.axis('off')
  
    ax2.imshow(f_xy_p, cmap='gray')
    ax2.set_title('Bước 1: Ảnh PxQ')
    ax2.axis('off')
 
    ax3.imshow(F_xy_p, cmap='gray')
    ax3.set_title('Bước 2: nhân -1 mũ x+y')
    ax3.axis('off')
 
    ax4.imshow(dft_hang, cmap='gray')
    ax4.set_title('Bước 3: Phổ tần số ảnh sau khi DFT')
    ax4.axis('off')

    ax5.imshow(H_uv, cmap='gray')
    ax5.set_title('Bước 4: Phổ tần số Bộ lọc')
    ax5.axis('off')

    ax6.imshow(G_uv, cmap='gray')
    ax6.set_title('Bước 5: Sau khi nhân DFT với ảnh sau khi lọc ')
    ax6.axis('off')

    ax7.imshow(idft_hang, cmap='gray')
    ax7.set_title('Bước 6.1: Thực hiện DFT ngược')
    ax7.axis('off')
  
    ax8.imshow(g_xy_p, cmap='gray')
    ax8.set_title('Bước 6.2: Phần thực sau IDFT nhân -1 mũ (x+y)')
    ax8.axis('off')
  
    ax9.imshow(g_xy, cmap='gray')
    ax9.set_title('Bước 7: Ảnh cuối cùng MxN')
    ax9.axis('off')

    plt.show()