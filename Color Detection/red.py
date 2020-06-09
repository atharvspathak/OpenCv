import cv2
import numpy as np 

cap=cv2.VideoCapture(0)

while 1:
	ret,frame=cap.read()
	hsv_frame =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	#red
	low_red =np.array([161,155,84])
	high_red =np.array([179,255,255])
	mask = cv2.inRange(hsv_frame,low_red,high_red)
	
		#yellow
	low_yellow =np.array([161,155,84])
	high_yellow =np.array([179,255,255])
	yellow_mask = cv2.inRange(hsv_frame,low_yellow,high_yellow)
	yellow=cv2.bitwise_and(frame,frame,mask=yellow_mask)

	#green
	low_green =np.array([25,52,72])
	high_green =np.array([0,255,0])
	green_mask = cv2.inRange(hsv_frame,low_green,high_green)
	green=cv2.bitwise_and(frame,frame,mask=green_mask)

	cv2.imshow('frame',frame)
	cv2.imshow('Red',mask)
	cv2.imshow('yellow',yellow)
	cv2.imshow('green',green)
	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
