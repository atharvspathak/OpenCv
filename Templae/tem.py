import cv2
import numpy as np 

img = cv2.imread("traffi2.jpg")
grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
template=cv2.imread("t3t.jpg",0)
w,h=template.shape[::-1]

res=cv2.matchTemplate(grey_img,template,cv2.TM_CCOEFF_NORMED)
print(res)
threshold=0.9;
loc = np.where(res>threshold)
print(loc)
for pt in zip(*loc[::-1]):
	cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)
cv2.imshow("Match",img)
key=cv2.waitKey(0)
if key == 27:
	cv2.destroyAllWindows()
