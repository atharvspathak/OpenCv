
import cv2
import numpy as np 

def nothing():
	pass

cap=cv2.VideoCapture(0)

#Create Trackbar
cv2.namedWindow("TrackBar")
cv2.createTrackbar("L-H","TrackBar",0,179,nothing)
cv2.createTrackbar("L-S","TrackBar",0,255,nothing)
cv2.createTrackbar("L-V","TrackBar",0,255,nothing)
cv2.createTrackbar("U-H","TrackBar",179,179,nothing)
cv2.createTrackbar("U-S","TrackBar",255,255,nothing)
cv2.createTrackbar("U-V","TrackBar",255,255,nothing)


while 1:
	ret,frame=cap.read()
	hsv_frame =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	lh = cv2.getTrackbarPos("L-H","TrackBar")
	ls = cv2.getTrackbarPos("L-S","TrackBar")
	lv = cv2.getTrackbarPos("L-V","TrackBar")
	uh = cv2.getTrackbarPos("U-H","TrackBar")
	us = cv2.getTrackbarPos("U-S","TrackBar")
	uv = cv2.getTrackbarPos("U-V","TrackBar")


	#red
	low_red =np.array([lh,ls,lv])
	high_red =np.array([uh,us,uv])
	red_mask = cv2.inRange(hsv_frame,low_red,high_red)
	red=cv2.bitwise_and(frame,frame,mask=red_mask)



	cv2.imshow('frame',frame)
	cv2.imshow('mask',red_mask)
	cv2.imshow('Red',red)
	
	k = cv2.waitKey(1)
	if k == 27:
			break

cv2.destroyAllWindows()
cap.release()
