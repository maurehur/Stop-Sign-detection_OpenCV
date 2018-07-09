import numpy as np
import cv2
import os
import sys
import subprocess

#---------------------------Documentation------------------------------
# http://answers.opencv.org/question/4368/traincascade-error-bad-argument-can-not-get-new-positive-sample-the-most-possible-reason-is-insufficient-count-of-samples-in-given-vec-file/#4474
# https://stackoverflow.com/questions/10863560/haar-training-opencv-assertion-failed
# http://www.answers.opencv.org/question/7141/about-traincascade-paremeters-samples-and-other/
# http://answers.opencv.org/question/7141/about-traincascade-paremeters-samples-and-other/
# https://stackoverflow.com/questions/16058080/how-to-train-cascade-properly/16058254
# http://docs.opencv.org/trunk/dc/d88/tutorial_traincascade.html
# http://answers.opencv.org/question/22964/opencv_traincascade-negative-samples-training-method/
# http://www.computer-vision-software.com/blog/2009/11/faq-opencv-haartraining/
# http://answers.opencv.org/question/39160/opencv_traincascade-parameters-explanation-image-sizes-etc/
#----------------------------------------------------------------------


reload(sys)  
sys.setdefaultencoding('utf8')

POS_FOLDER = sys.argv[1]

NEG_FOLDER = "neg"
DATA_FOLDER = "data"

BG_TXT_FILE = "bg.txt"
POS_TXT_FILE = "pos.txt"
VECTOR_FILE = "vector.vec"

PROCESS_POS_IMAGES = True

POS_SIZE = (25, 45)
NUM_OF_POS = 20
NUMPOS = int(NUM_OF_POS*0.80) 	# int((num_in_vec - numNeg)/(1 + (numStages-1)*(1-minHitRate))*0.95)
NUMNEG = 4000 					# The total number of negative(background) images
MEMORY_SIZE = str(1024*12)
NUM_STAGES = 12
MIN_HIT_RATE = 0.999
MAX_FALSE_ALARM_RATE = 0.4

def resize_background():
	for img in os.listdir(NEG_FOLDER):
		_, file_extension = os.path.splitext(img)
		file_extension = file_extension.lower()
		if file_extension.endswith("jpg") or file_extension.endswith("png"):
			try:
				# cv2.IMREAD_GRAYSCALE
				img_read = cv2.imread("%s/%s" % (NEG_FOLDER, img))
				# Resize
				resized_image = cv2.resize(img_read, (150, 100))
				cv2.imwrite("%s/%s" % (NEG_FOLDER, img), resized_image)
				
			except Exception as e:
				print(str(e))

def process_pos_image_and_get_pos_txt():
	if not os.path.exists(POS_TXT_FILE):
		with open(POS_TXT_FILE, 'a') as pos_txt:
			for img_file in os.listdir(POS_FOLDER):
				_, file_extension = os.path.splitext(img_file)
				file_extension = file_extension.lower()
				if file_extension.endswith("jpg") or file_extension.endswith("png"):
					try:
						# Read gray scale
						image_gray = cv2.imread("%s/%s" % (POS_FOLDER, img_file), cv2.IMREAD_GRAYSCALE)
						# Write line to pos.txt: object in position x: 0, y: 0,  w: 100, h: 150
						image_shape = image_gray.shape
						line = "%s/%s 1 0 0 %d %d\n" % (POS_FOLDER, img_file, image_shape[1], image_shape[0])
						pos_txt.write(line)
						### Image Processing: filter
						if PROCESS_POS_IMAGES:
							image_gray_blurred = cv2.bilateralFilter(image_gray, 5, 10, 10)
						cv2.imwrite("%s/%s" % (POS_FOLDER, img_file), image_gray_blurred)

					except Exception as e:
						print(str(e))
	else:
		print("Warning: %s already exists!!" % POS_TXT_FILE)

def create_neg():
	### create single bg.txt
	if not os.path.exists(BG_TXT_FILE):
		for img_file in os.listdir(NEG_FOLDER):	
			_, file_extension = os.path.splitext(img_file)
			file_extension = file_extension.lower()		
			if file_extension.endswith("jpg") or file_extension.endswith("png"):
				line = "%s/%s\n" % (NEG_FOLDER, img_file)
				with open(BG_TXT_FILE, 'a') as f:
					f.write(line)
	else:
		print("Warning: %s already exists!!" % BG_TXT_FILE)

def get_vector_file():
	if not os.path.exists(VECTOR_FILE):
		subprocess.check_call(["opencv_createsamples",
			"-info", POS_TXT_FILE,
			"-vec", VECTOR_FILE,
			"-num", str(NUM_OF_POS),
			"-w", str(POS_SIZE[0]),
			"-h", str(POS_SIZE[1])])
	else:
		print("Warning: %s already exists!!" % VECTOR_FILE)

def train_haar_cascade():
	if not os.path.exists(DATA_FOLDER):
		os.makedirs(DATA_FOLDER)

	# opencv_traincascade -data data -vec cropped.vec -bg bg.txt -numPos 1220 -numNeg 767 -numStages 12 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -w 48 -h 48
	subprocess.check_call(["opencv_traincascade",
		"-data", DATA_FOLDER,
		"-vec", VECTOR_FILE,
		"-bg", BG_TXT_FILE,
		"-numPos", str(NUMPOS),
		"-numNeg", str(NUMNEG),
		"-numStages", str(NUM_STAGES),
		"-precalcValBufSize", MEMORY_SIZE,
		"-precalcIdxBufSize", MEMORY_SIZE,
		"-minHitRate", str(MIN_HIT_RATE),
		"-maxFalseAlarmRate", str(MAX_FALSE_ALARM_RATE),
		"-w", str(POS_SIZE[0]),
		"-h", str(POS_SIZE[1])])


if __name__ == '__main__':
	resize_background()
	create_neg()
	process_pos_image_and_get_pos_txt()
	get_vector_file()
	# train_haar_cascade()