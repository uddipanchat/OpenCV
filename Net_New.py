import cv2
import numpy as np
import imutils

img = cv2.imread('/home/uddipan/Downloads/PC160889.JPG')

#Resizing the image
img = cv2.resize(img,(640,480))

# Changing the image into HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#splitting it into the H,S,V channels
h, s, v = cv2.split(hsv_img)

#Masking the image
lower_white = np.array([0,0,0])
upper_white = np.array([80,80,80])
mask = cv2.inRange(hsv_img, lower_white, upper_white)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img,img,mask=mask)

im2, contours, hier = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
	if 5000<cv2.contourArea(cnt)<6000:
		cv2.drawContours(im2,[cnt],-1,(128,255,255),3)
		cv2.drawContours(mask,[cnt],0,255,-1)

cv2.imshow('Image',img)
cv2.imshow('Contour',im2)
cv2.imshow('Mask',mask)
cv2.imshow('Res',res)

cv2.waitKey(0)
cv2.destroyAllWindows()