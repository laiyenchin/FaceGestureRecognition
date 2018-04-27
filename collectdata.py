# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

#* With "*" are things I added. so can be WRONG --LAI
#* imutils correct payh
#* /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/imutils/face_utils

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np






# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=200)

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

img_counter = 0



# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


	# detect faces in the grayscale frame
	# *rects should be the bounding box thing?
	rects = detector(gray, 0)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)  
		shape = face_utils.shape_to_np(shape)


	#draw rectangle
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		faceAligned = fa.align(frame, gray, rect)
		extract = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
	#*equalization
		#equ = cv2.equalizeHist(extract)
		clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(15,25))
		cl1 = clahe.apply(extract)
		smooth = cv2.bilateralFilter(cl1, 0, 20, 2)
		roi = smooth[0:90, 0:90]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		

	# show the face number
	cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)




	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	#for (i, j) in shape:
	#	cv2.circle(frame, (i, j), 1, (0, 0, 255), -1)
	
	# show the frame
	cv2.imshow("Frame", frame)
	#cv2.imshow("extract", extract)
	#cv2.imshow("equ", equ)
	cv2.imshow("roi", roi)
	key = cv2.waitKey(1) & 0xFF
 
	
	if key == ord("q"):
		break
    
    #* crop the rectangle and save in folder crops
   	elif key == ord("c"): 
   		#sub_face = frame[y: y + h, x: x + w]
   		filename = "face{}.0.png".format(img_counter)
   		img_counter += 1 
   		resize = cv2.resize(roi, (90, 90))
   		cv2.imwrite("data/"+filename, resize)
    
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()