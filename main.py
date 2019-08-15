import numpy as np
import cv2
import matplotlib.pyplot as plt

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



haar_cascade_face = cv2.CascadeClassifier('C:/py_home/github_master/posture-detection/data/haarcascades/haarcascade_frontalface_default.xml')

def detect_faces(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    gray_image = np.array(gray_image, dtype='uint8')

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)

    return image_copy


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #do face detection
    face_detect = detect_faces(haar_cascade_face, frame)

    # Display the resulting frame
    cv2.imshow('frame',face_detect)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

img = cv2.imread('test_image.jpg')

face_test = detect_faces(haar_cascade_face, img)
cv2.imshow('test', face_test)

cv2.waitKey()
