import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,50,50])
    upper_blue = np.array([130,255,255])
    lower_red = np.array([161, 155, 84])
    upper_red = np.array([179, 255, 255])
    lower_green = np.array([25, 52, 72])
    upper_green = np.array([102, 255, 255])


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange (hsv, lower_blue, upper_blue)
    bluecnts = cv2.findContours(mask.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[-2]
    mask1 = cv2.inRange (hsv, lower_red, upper_red)
    bluecnts_red = cv2.findContours(mask1.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[-2]
    mask2 = cv2.inRange (hsv, lower_green, upper_green)
    bluecnts_green = cv2.findContours(mask2.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(bluecnts)>0:
        blue_area = max(bluecnts, key=cv2.contourArea)
        (xg,yg,wg,hg) = cv2.boundingRect(blue_area)
        cv2.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(255,0,0),2)
        cv2.putText(frame,'Blue',(xg,yg-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),2)
    if len(bluecnts_red)>0:
        blue_area = max(bluecnts_red, key=cv2.contourArea)
        (xg,yg,wg,hg) = cv2.boundingRect(blue_area)
        cv2.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(0,0,255),2)
        cv2.putText(frame,'Red',(xg,yg-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
    if len(bluecnts_green)>0:
        blue_area = max(bluecnts_green, key=cv2.contourArea)
        (xg,yg,wg,hg) = cv2.boundingRect(blue_area)
        cv2.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(0,255,0),2)
        cv2.putText(frame,'Green',(xg,yg-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)


    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)

    k = cv2.waitKey(5)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
