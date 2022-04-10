import cv2
import matplotlib.pyplot as plt

img = cv2.imread('catnguong.tif',0)
img_equalized = cv2.equalizeHist(img)

fig = plt.figure(figsize=(16, 9))
(ax1, ax2), (ax3,ax4) = fig.subplots(2, 2)


ax1.imshow(img, cmap='gray')
ax1.set_title("ảnh gốc")


ax2.hist(img)
ax2.set_title("Histogram ảnh gốc")


ax3.imshow(img_equalized, cmap='gray')
ax3.set_title("ảnh cân bằng histogram")


ax4.hist(img_equalized)
ax4.set_title("Histogram ảnh cân bằng")

plt.show()