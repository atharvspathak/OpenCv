import cv2

face_cascade = cv2.CascadeClassifier('cascade.xml')
img = cv2.imread('ABC.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.1,4)

for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(2555,0,0),3)

cv2.imshow('img',img)
cv2.waitKey(3000)