import cv2
import numpy as np 
cap = cv2.VideoCapture(0)

while 1:
	_,img=cap.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	template=cv2.imread("pqr2.png",0)
	w,h=template.shape[::-1]

	res=cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
	print(res)
	threshold=0.8;
	loc = np.where(res>threshold)
	print(loc)
	for pt in zip(*loc[::-1]):
		cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)


	cv2.imshow("Match",img)
	key=cv2.waitKey(1)
	if key == 27:
		break
cap.release()
cv2.destroyAllWindows()
