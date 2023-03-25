import cv2
import datetime

today = datetime.datetime.now()

day_folder = "../logs/" + str(today.day) + "-" + str(today.month) + "-" + str(today.year) + "/"
path = day_folder + "test1_tv.avi"

cap = cv2.VideoCapture(path)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame", frame)
    else:
        print("frame not found")
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()



