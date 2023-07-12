import cv2
import imutils
cam = cv2.VideoCapture(0)
firstframe= None
area = 500
while True:
    img = cam.read()[1]
    img = imutils.resize(img, width=1000)
    text="no moving object detected"
    grayimg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gaussianimg = cv2.GaussianBlur(grayimg, (21,21),0)
    if firstframe is None:
        firstframe =gaussianimg
        continue
    imgdiff = cv2.absdiff(firstframe,gaussianimg)
    threshimg = cv2.threshold(imgdiff, 25,255,cv2.THRESH_BINARY)[1]
    threshimg = cv2.dilate(threshimg,None,iterations=2)
    contours = cv2.findContours(threshimg.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    count=0
    for i in contours:
        if cv2.contourArea(i) < area:
            continue
        (x,y,w,h) = cv2.boundingRect(i)
        count +=1
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        text="{} Moving object detected".format(count)

    cv2.putText(img,text,(10,30),cv2.FONT_HERSHEY_PLAIN,1,(100,20,30),2)
    cv2.imshow("cam feed",img)

    key = cv2.waitKey(1)
    if key is ord("q"):
        break
cam.release()
cv2.destroyAllWindows()

