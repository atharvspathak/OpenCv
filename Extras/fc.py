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

def empty(a):
    pass

def init():
	cap = cv2.VideoCapture(0)
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	tracBar()
	cv2.resizeWindow("Result",640,580)
	cv2.createTrackbar("Scale","Result",400,1000,empty)
	cv2.createTrackbar("Neig","Result",8,50,empty)
	cv2.createTrackbar("Min Area","Result",0,100000,empty)
	cv2.createTrackbar("Brightness","Result",180,255,empty)

	while 1:
		
		ret,img = cap.read()
		cameraBrightness = cv2.getTrackbarPos("Brightness", "Result")
    		cap.set(10, cameraBrightness)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		scaleVal =1 + (cv2.getTrackbarPos("Scale", "Result") /1000)
   		neig=cv2.getTrackbarPos("Neig", "Result")
		faces = face_cascade.detectMultiScale(gray,1.1,4)
		for x,y,w,h in faces:
			area = w*h
     			minArea = cv2.getTrackbarPos("Min Area", "Result")
			if area > minArea:
		    		cv2.rectangle(img,'Atharv',(x,y),(x+w,y+h),(255,0,0),1)
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



