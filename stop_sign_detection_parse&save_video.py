import cv2
import numpy as np

def define_region(image):
	### black region
	if (len(image.shape) == 3): 
		height, length, _ = image.shape
	else:
		height, length = image.shape

	# Vertices array of the unwanted area --> bottom 1/3 of the picture
	region = [np.array([(0, height), (0, height//3), (length, height//3), (length, height)])]

	return region

def crop_frame(image):
	if (len(image.shape) == 3): 
		height, length, _ = image.shape
	else:
		height, length = image.shape
	return image[0: height//3*2, 0: length]

def mask_image(image, vertices):
	# fill unwanted area with zeros
	cv2.fillPoly(image, vertices, 0)
	return image

if __name__ == '__main__':
	# video path
	filepath = 'stop_sign_videos\7.mp4'
	# load classifier
	stop_sign_cascade = cv2.CascadeClassifier('stop_sign_classifier_2.xml')

	cap = cv2.VideoCapture(filepath)

	# save frames to video
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	video = cv2.VideoWriter('stop_sign_videos\out\6.avi', fourcc, 20, (1080, 1920))

	cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame', 360, 640)

	while (cap.isOpened()):
		ret, frame = cap.read()

		if (ret):
			cropped_frame = crop_frame(frame)
			
			img_filter = cv2.GaussianBlur(cropped_frame, (5, 5), 0)
			gray_filered = cv2.cvtColor(img_filter, cv2.COLOR_BGR2GRAY)

			stop_signs = stop_sign_cascade.detectMultiScale(gray_filered, scaleFactor=1.05, minNeighbors=15, minSize=(30, 30))

			for (x,y,w,h) in stop_signs:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 3)

			video.write(frame)

			cv2.imshow('frame', frame)
			cv2.waitKey(1)
		else:
			break

		k=cv2.waitKey(1) & 0xFF
		if k==27:
			break

	video.release()
	cap.release()
	cv2.destroyAllWindows()