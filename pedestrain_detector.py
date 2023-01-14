import cv2

trained_fullbody_data = cv2.CascadeClassifier('fullbody.xml')

trained_cars_data = cv2.CascadeClassifier('cars.xml')

detect = cv2.VideoCapture('dtet.mp4') 
 
while True:
    successful_frame_read, frame = detect.read()
    gray_detect = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fullbody = trained_fullbody_data.detectMultiScale(gray_detect)
    cars = trained_cars_data.detectMultiScale(gray_detect)

    for (x, y, w, h) in fullbody:
            cv2.rectangle(frame, (x+2, y+2), (x+w, y+h), (225, 225, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (225, 225, 0), 2)
    for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 0), 2)        

    cv2.imshow('swagalomo', frame)
    end = cv2.waitKey(1)     

    if end == 81 or end == 113:
      break;

detect.release()  