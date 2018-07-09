import cv2
import numpy as np
from PIL import ImageGrab

# Load haar cascade xml file
stop_sign_cascade = cv2.CascadeClassifier('stop_sign_classifier_2.xml')

while True:
	### Take screen shots
	# define screen shot area 
	# box = (960, 0, 1920, 720)
	box=(0, 0, 960, 720) # smaller screen
	image = ImageGrab.grab(box) # bbox specifies a region (bbox= x, y, w, h). x, y --> origin. w, h --> width, height
	# Convert to numpy array
	image_np = np.array(image) 
	# Convert to BGR(openCV format) and then to gray scale
	image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
	gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
	# Apply Gaussian filter
	gray_filered = cv2.GaussianBlur(gray, (5, 5), 0)
	
    # Detection
	stop_signs = stop_sign_cascade.detectMultiScale(gray_filered, scaleFactor=1.05, minNeighbors=5, minSize=(5, 5))
	
	print(len(stop_signs))
	# Draw rectangels
	for (x,y,w,h) in stop_signs:
		cv2.rectangle(image_np, (x, y), (x+w, y+h), (255, 255, 0), 2)

	cv2.namedWindow("screenshot", cv2.WINDOW_NORMAL)
	cv2.resizeWindow('screenshot', 640, 480)
	cv2.imshow('screenshot',image_np)

	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break

# cap.release()
cv2.destroyAllWindows()