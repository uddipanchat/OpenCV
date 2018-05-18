'''
    Net.py
    Purpose: Detect underwater Fishing Net Patterns and Detect any irregularities

    @author Uddipan Chatterjee
    Version: 1.0

''' 

import cv2
import numpy as np
import math

class ShapeDetection():

	def __init__(self, img):
		self.img = img
    
	def Preprocess(self):
		lower_bound = np.array([130,130,130], dtype=np.uint8)
		upper_bound = np.array([255,255,255], dtype=np.uint8)
		mask = cv2.inRange(self.img,lower_bound,upper_bound)
		cv2.imshow("mask", mask)




img = cv2.imread("/home/uddipan/Downloads/hole1.JPG") 
small_img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
ShapeDetection = ShapeDetection(small_img)
ShapeDetection.Preprocess()

cv2.imshow("Image", small_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
