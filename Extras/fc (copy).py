import cv2
import numpy as np

def nothing():
	pass


def colorDetect(img):
	hsv_frame =cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

	lh = cv2.getTrackbarPos("L-H","TrackBar")
	ls = cv2.getTrackbarPos("L-S","TrackBar")
	lv = cv2.getTrackbarPos("L-V","TrackBar")
	uh = cv2.getTrackbarPos("U-H","TrackBar")
	us = cv2.getTrackbarPos("U-S","TrackBar")
	uv = cv2.getTrackbarPos("U-V","TrackBar")

	low_red =np.array([lh,ls,lv])
	high_red =np.array([uh,us,uv])
	red_mask = cv2.inRange(hsv_frame,low_red,high_red)
	red=cv2.bitwise_and(img,img,mask=red_mask)

	

	cv2.imshow('mask',red_mask)
	cv2.imshow('Red',red)
	
def tracBar():
	cv2.namedWindow("TrackBar")
	cv2.createTrackbar("L-H","TrackBar",0,179,nothing)
	cv2.createTrackbar("L-S","TrackBar",0,255,nothing)
	cv2.createTrackbar("L-V","TrackBar",0,255,nothing)
	cv2.createTrackbar("U-H","TrackBar",179,179,nothing)
	cv2.createTrackbar("U-S","TrackBar",255,255,nothing)
	cv2.createTrackbar("U-V","TrackBar",255,255,nothing)

def init():
	cap = cv2.VideoCapture(0)
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	tracBar()

	while 1:
		ret,img = cap.read()
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		faces = face_cascade.detectMultiScale(gray,1.1,4)
		for x,y,w,h in faces:
	    		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
			roi_gray = gray[y:y+h, x:x+w] 
			roi_color = img[y:y+h, x:x+w] 
			cv2.imshow('croped',roi_color)
		
		colorDetect(roi_color)
		cv2.imshow('img',img)
		k = cv2.waitKey(30) & 0xff
		if k == 27: 
			break

	cap.release()
	cv2.destroyAllWindows() 

init()



