import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('test_image.jpg')

test_image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
test_image_gray = np.array(test_image_gray, dtype='uint8')

#plt.imshow(test_image_gray, cmap='gray')

#cv2.imshow('face detection',test_image_gray)



haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor=1.3, minNeighbors=5)

print('Faces found:', len(faces_rects))

for(x,y,w,h) in faces_rects:
    cv2.rectangle(test_image_gray, (x,y), (x+w, y+h), (0, 255, 0), 2)

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cv2.imshow('test',convertToRGB(test_image_gray))

cv2.waitKey()
