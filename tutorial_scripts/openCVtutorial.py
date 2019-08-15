import numpy as np
import cv2
import matplotlib.pyplot as plt

#generate black square to display
img = np.zeros(shape=(512,512,3))
plt.imshow(img)

#drawing shapes on images
#cv2.shape(line,rectangle, etc)(image, pt1,pt2, color, thickness)
linediag_red = cv2.line(img,(0,0),(511,511),(255,0,0),5)
plt.imshow(linediag_red)

linediag_green = cv2.line(img,(0,511),(511,0),(0,255,0),8)
plt.imshow(linediag_green)

#drawing blue rectangle
rectangle = cv2.rectangle(img,(384,0),(510,128),(0,0,255),5)
plt.imshow(rectangle)

#circle, need center and radius
circle_blue = cv2.circle(img,(447,63),63,(0,0,255),-1)
plt.imshow(circle_blue)

#writing on images
font = cv2.FONT_HERSHEY_SIMPLEX
text = cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
plt.imshow(text)

cv2.imshow('tutorial',img)
