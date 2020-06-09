import cv2
import numpy as np
################################################################
path = 'haarcascade_frontalface_default.xml'  # PATH OF THE CASCADE
cameraNo = 0                       # CAMERA NUMBER
objectName = 'Prince'       # OBJECT NAME TO DISPLAY
frameWidth= 640                     # DISPLAY WIDTH
frameHeight = 480                  # DISPLAY HEIGHT
color= (255,0,255)
scaleVal=1.1
#################################################################

	

def nothing():
	pass

def empty(a):
    pass

def redDetect(img):
	cv2.namedWindow("Red")
	cv2.createTrackbar("L-H","Red",0,179,nothing)
	cv2.createTrackbar("L-S","Red",0,255,nothing)
	cv2.createTrackbar("L-V","Red",0,255,nothing)
	cv2.createTrackbar("U-H","Red",179,179,nothing)
	cv2.createTrackbar("U-S","Red",255,255,nothing)
	cv2.createTrackbar("U-V","Red",255,255,nothing)
	hsv_frame =cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	Threshold=0.01
	lh = cv2.getTrackbarPos("L-H","Red")
	ls = cv2.getTrackbarPos("L-S","Red")
	lv = cv2.getTrackbarPos("L-V","Red")
	uh = cv2.getTrackbarPos("U-H","Red")
	us = cv2.getTrackbarPos("U-S","Red")
	uv = cv2.getTrackbarPos("U-V","Red")

	low_red =np.array([lh,ls,lv])
	high_red =np.array([uh,us,uv])
	red_mask = cv2.inRange(hsv_frame,low_red,high_red)
	bluecnts_red = cv2.findContours(red_mask.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[-2]

	if len(bluecnts_red)>0:
       	 blue_area = max(bluecnts_red, key=cv2.contourArea)
       	 (xg,yg,wg,hg) = cv2.boundingRect(blue_area)
       	 cv2.rectangle(img,(xg,yg),(xg+wg, yg+hg),(0,0,255),2)
       	 cv2.putText(img,'Red',(xg,yg-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
	
	rate=np.count_nonzero(red_mask)/(30*90)
	print(rate,"r")
	

	cv2.imshow('maskr',red_mask)
	cv2.imshow('Red',img)

def greenDetect(img):
	cv2.namedWindow("Green")
	cv2.createTrackbar("L-H","Green",0,179,nothing)
	cv2.createTrackbar("L-S","Green",0,255,nothing)
	cv2.createTrackbar("L-V","Green",0,255,nothing)
	cv2.createTrackbar("U-H","Green",179,179,nothing)
	cv2.createTrackbar("U-S","Green",255,255,nothing)
	cv2.createTrackbar("U-V","Green",255,255,nothing)
	hsv_frame =cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	Threshold=0.01
	lh = cv2.getTrackbarPos("L-H","Green")
	ls = cv2.getTrackbarPos("L-S","Green")
	lv = cv2.getTrackbarPos("L-V","Green")
	uh = cv2.getTrackbarPos("U-H","Green")
	us = cv2.getTrackbarPos("U-S","Green")
	uv = cv2.getTrackbarPos("U-V","Green")

	low_green =np.array([lh,ls,lv])
	high_green =np.array([uh,us,uv])
	green_mask = cv2.inRange(hsv_frame,low_green,high_green)
	green=cv2.bitwise_and(img,img,mask=green_mask)
	
	rate=np.count_nonzero(green_mask)/(30*90)
	print(rate,"g")
	

	cv2.imshow('maskg',green_mask)
	cv2.imshow('Green',green)

def yellowDetect(img):
	cv2.namedWindow("yellow")
	cv2.createTrackbar("L-H","yellow",0,179,nothing)
	cv2.createTrackbar("L-S","yellow",0,255,nothing)
	cv2.createTrackbar("L-V","yellow",0,255,nothing)
	cv2.createTrackbar("U-H","yellow",179,179,nothing)
	cv2.createTrackbar("U-S","yellow",255,255,nothing)
	cv2.createTrackbar("U-V","yellow",255,255,nothing)
	hsv_frame =cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	Threshold=0.01
	lh = cv2.getTrackbarPos("L-H","yellow")
	ls = cv2.getTrackbarPos("L-S","yellow")
	lv = cv2.getTrackbarPos("L-V","yellow")
	uh = cv2.getTrackbarPos("U-H","yellow")
	us = cv2.getTrackbarPos("U-S","yellow")
	uv = cv2.getTrackbarPos("U-V","yellow")

	low_yellow =np.array([lh,ls,lv])
	high_yellow =np.array([uh,us,uv])
	yellow_mask = cv2.inRange(hsv_frame,low_yellow,high_yellow)
	yellow=cv2.bitwise_and(img,img,mask=yellow_mask)
	
	rate=np.count_nonzero(yellow_mask)/(30*90)
	print(rate,"y")
	

	cv2.imshow('masky',yellow_mask)
	cv2.imshow('yellow',yellow)
	

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)



# CREATE TRACKBAR
cv2.namedWindow("Result")
cv2.resizeWindow("Result",frameWidth,frameHeight+100)
cv2.createTrackbar("Scale","Result",1,9,empty)
cv2.createTrackbar("Neig","Result",8,50,empty)
cv2.createTrackbar("Min Area","Result",0,100000,empty)
cv2.createTrackbar("Brightness","Result",180,255,empty)

# LOAD THE CLASSIFIERS DOWNLOADED
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # SET CAMERA BRIGHTNESS FROM TRACKBAR VALUE
    cameraBrightness = cv2.getTrackbarPos("Brightness", "Result")
    cap.set(10, cameraBrightness)

    # GET CAMERA IMAGE AND CONVERT TO GRAYSCALE
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # DETECT THE OBJECT USING THE CASCADE
    scaleVal =1.1+((cv2.getTrackbarPos("Scale", "Result"))/100)
    neig=cv2.getTrackbarPos("Neig", "Result")
    objects = cascade.detectMultiScale(gray,scaleVal, neig)

    # DISPLAY THE DETECTED OBJECTS
    for (x,y,w,h) in objects:
        area = w*h
        minArea = cv2.getTrackbarPos("Min Area", "Result")
        if area >minArea:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
            cv2.putText(img,objectName,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            roi_color = img[y:y+h, x:x+w]
	    redDetect(roi_color)
	    greenDetect(roi_color)
	    yellowDetect(roi_color)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break


