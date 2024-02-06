from ultralytics import YOLO
import cv2
import cvzone
import math


#load model
model = YOLO("Pistol_Detection.pt")
cap = cv2.VideoCapture("Pistol.mp4")

# label on my dataset
names=["Pistol"]

while True:
    # read the image
    ret, image = cap.read()
    if ret:
        # resize the frame
        image=cv2.resize(image,(700,600))
        # result of model
        result = model(image, stream=True)
        for i in result:
            boxs = i.boxes
            for box in boxs:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w,h=x2-x1,y2-y1
                # draw the rectangle
                cvzone.cornerRect(image, (x1, y1, w, h))
                conf=math.ceil(box.conf[0]*100)/100
                cls=int(box.cls[0])
                print(conf,names[cls])
                # Put text on frame
                cvzone.putTextRect(image,f"{names[cls]}",(x1,y1-10),scale=2,colorB="blue")

        # show the frame
        cv2.imshow("Pistol_Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
