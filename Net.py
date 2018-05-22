'''
    Net.py
    Purpose: Detect underwater Fishing Net Patterns and Detect any irregularities

    @author Uddipan Chatterjee
    Version: 1.0

''' 

import cv2
import numpy as np
import math
import imutils

class ShapeDetection():

	def __init__(self, img):
		self.img = img
    
	def Preprocess(self):
		lower_bound = np.array([80,80,80], dtype=np.uint8)
		upper_bound = np.array([255,255,255], dtype=np.uint8)
		mask = cv2.inRange(self.img,lower_bound,upper_bound)
		mask = 255 - mask
		output = cv2.bitwise_and(self.img, self.img, mask=mask)
		#im1 = cv2.cvtColor( self.img, cv2.COLOR_RGB2GRAY )
		#mask_inv = 	255 - mask
		#output = mask_inv*im1
		#cv2.imshow("Output",output)

		#output = cv2.bitwise_and(img, img, mask=mask)

		ret,thresh = cv2.threshold(mask, 40, 255, 0)
		#cv2.imshow("theshold",thresh)
		im2, contours, hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		c = max(contours, key = cv2.contourArea)
		cv2.drawContours(output, [c], -1, 255, 3)
		cv2.imshow("Final_Image",output)

		#if len(contours) != 0:
		
'''
		for cnt in contours:
			if cv2.contourArea(cnt) > 6000:
				cv2.drawContours(output, [cnt], -1, 255, 3)
'''

				#x,y,w,h = cv2.boundingRect(c)
				#cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
		
		#cv2.drawContours(im2,contours, -1, (255,0,0), 3)
		#cv2.imshow("Final_Image",im2)
		
'''
		for cnt in contours:
			if cv2.contourArea(cnt) > 6000:
				#cv2.drawContours(im2,[cnt],-1,(0,255,0),3)
				cv2.drawContours(mask,[cnt],0,(0,0,255),3)
'''		
		
        
		
		#cv2.imshow("Hole",c)

'''
		contours = contours[0] if imutils.is_cv2() else contours[1]
		c = max(contours, key=cv2.contourArea)
		cv2.drawContours(im2, [c], 1, (0, 255, 255), 2)
		
'''

       

cap = cv2.VideoCapture(0)


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")


while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True::
		cv2.imshow('Frame',frame)
		small_img = cv2.resize(frame, (640,480))

		#
		#bordersize=10
		#small_img=cv2.copyMakeBorder(small_img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )

		#Create the identity filter, but with the 1 shifted to the right!
		kernel = np.zeros( (9,9), np.float32)
		kernel[4,4] = 2.0   #Identity,times two! 

		#Create a box filter:
		boxFilter = np.ones( (9,9), np.float32) / 81.0

		#Subtract the two:
		kernel = kernel - boxFilter


		#Note that we are subject to overflow and underflow here...but I believe that
		# filter2D clips top and bottom ranges on the output, plus you'd need a
		# very bright or very dark pixel surrounded by the opposite type.

		custom = cv2.filter2D(small_img, -1, kernel)
		#cv2.imshow("Sharpen", custom)
		ShapeDetection = ShapeDetection(custom)
		ShapeDetection.Preprocess()

		#cv2.imshow("Image", small_img)
		
	# Press Q on keyboard to  exit
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break
		# Break the loop
	else: 
		break

cv2.waitKey(0)

# When everything done, release the video capture object
cap.release()
 
cv2.destroyAllWindows()


#img = cv2.imread("/home/uddipan/Downloads/P5080325.JPG") 
