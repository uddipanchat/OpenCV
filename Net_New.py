import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from random import *




def Find_Index(contours,max_contour):
	index = 0
	max = np.array(max_contour)
	for cnt in contours:
		if np.array_equal(np.array(cnt),max):
			return(index)
		else:
			index = index + 1



def Find_True_Contour(contours,height,width):

	false_detection = False
	c = max(contours, key=cv2.contourArea)
	for i in range(0,(len(c)-1)):
		if (c[i][0][0] == 0) or (c[i][0][0] == height) or ((c[i][0][1] == 0) or (c[i][0][1] == width)):
			false_detection = True
			break;
	if false_detection == True:
		index = Find_Index(contours,c)
		#print("No of contours",len(contours))
		#print("Index:",index)
		if (len(contours) >  1):
			del contours[index]
			return(Find_True_Contour(contours,height,width))
		else:
			false_detection = False
	if false_detection == False:
		#print(c)
		return(c)		



def DetectHoles(img):
	lower_bound = np.array([80,80,80], dtype=np.uint8)
	upper_bound = np.array([255,255,255], dtype=np.uint8)
	mask = cv2.inRange(img,lower_bound,upper_bound)
	mask = 255 - mask
	output = cv2.bitwise_and(img, img, mask=mask)
		

	ret,thresh = cv2.threshold(mask, 40, 255, 0)
	height, width = thresh.shape
	#cv2.imshow("theshold",thresh)
	im2, contours, hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	#con = np.array(contours)
	#print(con.shape)
	print("Initial no of contours:", len(contours))
	#c1=[]
	c = Find_True_Contour(contours,height,width)
	#print(c)
	cv2.drawContours(output,[c],-1 , 255, 1)
	image_no = randint(0,100)
	cv2.imwrite('/home/uddipan/Documents/NetDetection/Detected_Images/NetDefect'+ str(image_no) +'.png',output)
	return(output)
'''	
	c = max(contours, key = cv2.contourArea)
	#print(contours.shape)
	#cv2.drawContours(output, [c], -1, (0,0,255), 3)
	#return(output)
	#cv2.imshow("Final_Image",output)
	


	false_detection = False
	for i in range(0,(len(c)-1)):
		if (c[i][0][0] == 0) or (c[i][0][0] == height) or ((c[i][0][1] == 0) or (c[i][0][1] == width)):
			false_detection = True
			break;

	if false_detection == True:
		index = Find_Index()
		cv2.drawContours(output, [c], -1, (0,0,255), 3)
		return(output)
	else:
		cv2.drawContours(output,[c], -1 , 255, 1)
		image_no = randint(0,100)

'''




def SharpenImage(img):
	
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

	custom = cv2.filter2D(img, -1, kernel)

	return(custom)


def WriteVideo(frame):

	height, width, channels = frame.shape
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))
	out.write(frame)









cap = cv2.VideoCapture('/home/uddipan/Downloads/fish_net.mpg')


writer = None


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")


while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True:

		# Down-size the image

		small_img = cv2.resize(frame, (640,480))

		# Sharpen the image

		sharpe_image = SharpenImage(small_img)

		# Detect holes with contours

		new_frame = DetectHoles(sharpe_image)

		# Write new frame to a video file

		#WriteVideo(new_frame)
        if writer is None:
        	height, width, channels = new_frame.shape
        	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height), True)

		
		out.write(new_frame)
		cv2.imshow("Input", small_img)
		cv2.imshow("Output", new_frame)
		key = cv2.waitKey(1) & 0xFF


		if key == ord("q"):
			break







cv2.waitKey(0)

# When everything done, release the video capture object
cap.release()
 
cv2.destroyAllWindows()

writer.release()





		#
		#bordersize=10
		#small_img=cv2.copyMakeBorder(small_img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )

		
