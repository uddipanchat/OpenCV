import cv2
import numpy as np




def Find_Index(contours,max_contour):
	index = 0
	max = np.array(max_contour)
	for cnt in contours:
		if np.array_equal(np.array(cnt),max):
			return(index)
		else:
			index = index + 1


img=cv2.imread("/home/uddipan/Downloads/P8060006.JPG")

img_small = cv2.resize(img, (640,480))

#img_gray=cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

kernel = np.zeros( (9,9), np.float32)
kernel[4,4] = 2.0   #Identity,times two! 

	#Create a box filter:
boxFilter = np.ones( (9,9), np.float32) / 81.0

	#Subtract the two:
kernel = kernel - boxFilter


	#Note that we are subject to overflow and underflow here...but I believe that
	# filter2D clips top and bottom ranges on the output, plus you'd need a
	# very bright or very dark pixel surrounded by the opposite type.

custom = cv2.filter2D(img_small, -1, kernel)

bordersize=4
custom=cv2.copyMakeBorder(custom, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] ) 
height, width, channel = custom.shape
#print(height)

lower_bound = np.array([80,80,80], dtype=np.uint8)
upper_bound = np.array([255,255,255], dtype=np.uint8)
mask = cv2.inRange(custom,lower_bound,upper_bound)
mask = 255 - mask
#cv2.imshow("Mask0",mask)
output = cv2.bitwise_and(custom, custom, mask=mask)
		
#thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
ret,thresh = cv2.threshold(mask, 40, 255, 0)
#cv2.imshow("theshold",thresh)
c1 = []
im2, contours, hier = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
	if 4000 <cv2.contourArea(cnt):
		c1.append(cnt)

		




#print(contours[1])
#print(c1)
print("No of contours",len(contours))
#c = max(contours, key = cv2.contourArea)
#index = Find_Index(contours,c)
c = contours[len(contours) - 3]
#print(index)
#del contours[index]

#c = max(contours, key = cv2.contourArea)
#c2 = np.array(contours)
#ind = contours.index(c2.any(c))
#c1 = ''.join([str(x) for x in c] )
#str1 = ''.join(str(e) for e in contours)
#ind = str1.index(c1)
#print(ind)
#print(len(contours))
flag = 0 
for i in range(0,(len(c)-1)):
	if (c[i][0][0] == bordersize) or (c[i][0][0] == height) or ((c[i][0][1] == bordersize) or (c[i][0][1] == width)):
		flag = 1

		#del contours[ind]
		break

c1 = max(contours,key = cv2.contourArea)


#print(count)
#print(len(c))

#print "".join([str(x) for x in c] )

M = cv2.moments(c)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

#print(c)

cv2.drawContours(output, [c], -1, 255, 3)






cv2.imshow("Mask",mask)

cv2.imshow("Final_Image",output)

cv2.waitKey(0)

cv2.destroyAllWindows()
