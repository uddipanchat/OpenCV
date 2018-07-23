import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from random import *


def main(img):
	
	# Down-size the image

	small_img = cv2.resize(img, (640,480))

        # Center the image around the middle
        
        height , width, channels = small_img.shape
        cent_img = centerImage(small_img,width,height)
      

	# Sharpen the image

	#sharpe_image = SharpenImage(small_img)

	# Blur the image

        blur_image = BlurImage(cent_img)

        # Detect holes with contours

	new_frame = DetectHoles(blur_image)

	return(new_frame)



# Process everything on the center of the image
def centerImage(img,w,h):
    height, width, channels = img.shape
    upper_left = (int(width/2-w/2), int(height/2-h/2))
    bottom_right = (int(width/2+w/2), int(height/2+h/2))

    rect_img = img[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    return rect_img

# Median Blur 
def BlurImage(img):
	
	median = cv2.medianBlur(img,5)
	return(median)


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
		return(false_detection,c)		

'''

def Find_True_Contour(contours,height,width):

    false_detection = False
    for i in range((len(contours)-1),0,-1):
        false_detection = False
        c = contours[i]
        area = cv2.contourArea(c)
        # Check area of contour, if too small discard the frame
        if area < 20:
            print("area too small")
            false_detection = True
            return(false_detection,c)
          

        else:
            for i in range(0,(len(c)-1)):
                if (c[i][0][0] == 0) or (c[i][0][0] == height) or ((c[i][0][1] == 0) or (c[i][0][0] == width)):
                    false_detection = True
                    break;
       
        if false_detection == False:
            return(false_detection,c)
'''           


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
	#print("Initial no of contours:", len(contours))
	#c1=[]
	false_detection,c = Find_True_Contour(contours,height,width)
        if false_detection == False:
            #print("net hole found")
            cv2.drawContours(output,[c],-1 , 255, 1)
            image_no = randint(0,100)
            cv2.imwrite('/home/uddipan/Documents/NetDetection/Detected_Images/NetDefect'+ str(image_no) +'.png',output)
	return(output)


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




'''




cap = cv2.VideoCapture('/home/uddipan/Documents/Aquaai/bagfile_images/output.mpg')


writer = None


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")


while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True:

		
		# Write new frame to a video file

		#WriteVideo(new_frame)
        
		new_frame = main(frame)

        if writer is None:
        	height, width, channels = new_frame.shape
        	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height), True)

		
		out.write(new_frame)
		#cv2.imshow("Input", small_img)
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

		'''
 
