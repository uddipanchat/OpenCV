#!/usr/bin/env python

import numpy as np 
import cv2
import sys

import roslib
import rospy

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import Net_New


VERBOSE = False


class net_detect:

	def __init__(self):
		''' Initialize ROS publisher and Ros Subscriber '''
		
		#Publish the topic
		self.image_pub = rospy.Publisher("/processed/image_raw",Image, queue_size= 1)

		self.bridge = CvBridge()

		#Subscribe to topic

		self.subscriber = rospy.Subscriber("/camera/image_raw",Image, self.callback, queue_size = 1 )

		if VERBOSE:

			print "subscribed to /camera/image_raw"


	def callback(self,data):

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		rows,cols,channels = cv_image.shape
		if cols > 60 and rows > 60 :
			cv2.circle(cv_image, (50,50), 10, 255)
		
		cv_image = Net_New.main(cv_image)
        

	    #cv2.imshow("Image window", cv_image)
	    #cv2.waitKey(3)

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e)



def main(args):
	''' Initializes and cleans up ROS node '''
	
	rospy.init_node('net_detect', anonymous=True)
	Net_obj = net_detect()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print ""
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
