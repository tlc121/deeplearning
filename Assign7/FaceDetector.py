import numpy as np
import cv2
import time

#get the data which is already trained
cascade_path = "/Users/xavier0121/Desktop/haarcascade_frontalface_default.xml"
cascade_eye_path = "/Users/xavier0121/Desktop/haarcascade_eye.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascade_path)
eye_cascade = cv2.CascadeClassifier(cascade_eye_path)

#get the camera
webcam = cv2.VideoCapture(0)

#set the size of the camera.
#http://opencv-python-tutroals.readthedocs.io/en/
webcam.set(3, 600)
webcam.set(4, 800)

#set a timer
s_time = time.time()

#30 frames per second
for i in range(480):
    time_now = time.time()
    if time_now - s_time > 1/30:
        i += 1
        s_time = time_now
        #screen shot
        ret, frame = webcam.read()
        #convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detect the face
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # get gray value of face area.
            roi_gray = gray[y:y + h, x:x + w]

            # get color value of face area.
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (eye_x, eye_y, eye_w, eye_h) in eyes:
                cv2.rectangle(roi_color, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 0, 0), 2)
                cv2.imshow("camera", frame)
        cv2.imwrite("/Users/xavier0121/Desktop/Video/s" + str(i) + ".jpg", frame, [int(cv2.IMWRITE_PNG_COMPRESSION),20])
        key = cv2.waitKey(1)

        #shut down the program when pressing esc
        if key % 256 == 27:
            break

webcam.release()
cv2.destroyAllWindows()