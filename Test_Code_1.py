import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/home/uddipan/Downloads/PC160889.JPG")
#if img == None:
#	raise exception("Failed to load image!")
img = cv2.resize(img,(640,480))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_filtered = cv2.inRange(img_gray, 30 , 255)

gray_filtered_inv = 255 - gray_filtered

cv2.imshow("Gray_Img", img_gray)

cv2.imshow("Filtered_img", gray_filtered_inv)

cv2.waitKey(0)
cv2.destroyAllWindows()




'''

median = cv2.medianBlur(img,5)


ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


h, s, v = cv2.split(img_hsv)
cv2.imwrite('hue.png', img_hsv[:,:,0])
cv2.imwrite('sat.png', img_hsv[:,:,1])
cv2.imwrite('val.png', img_hsv[:,:,2])

cv2.imshow("HSV_IMG",img_hsv)


cv2.waitKey(0)

cv2.destroyAllWindows()
'''
