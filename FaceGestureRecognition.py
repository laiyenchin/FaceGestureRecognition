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





#faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
#cam=cv2.VideoCapture(0);
rec=cv2.createFisherFaceRecognizer();
rec.load('reader_all.yml')
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,1,1,0,1)

#--------------------------------
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


		id,conf=rec.predict(roi.copy())
		print conf
		#reader confidence
		if(400>=conf>=10):
		#player confidence
		#if(300>=conf>=100):

			#if(id==0):
			#	id="none"
			if(id==1):
				id="bigeyes"
				print id
			elif(id==2):
				id="squint"
				print id
			elif(id==3):
				id="lookup"
				print id
			elif(id==4):
				id="lookdown"
				print id
			elif(id==5):
				id="lookleft"
				print id
			elif(id==6):
				id="lookright"
				print id
			elif(id==7):
				id="blink"
				print id
		else:
			id="natural"
			print id

		cv2.cv.PutText(cv2.cv.fromarray(frame),str(id),(x,y+h),font,(0, 255, 0));

	cv2.imshow("Frame", frame)
	#cv2.imshow("equ", equ)
	#cv2.imshow("smooth", smooth)
	#cv2.imshow("roi", roi)
	key = cv2.waitKey(1) & 0xFF
 
	
	if key == ord("q"):
		break

#--------------------------------

cv2.destroyAllWindows()
vs.stop()


#while(True):
	#ret,img=cam.read();
	#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#faces=faceDetect.detectMultiScale(gray,1.3,5);
	#for(x,y,w,h) in faces:
	#	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	#	id,conf=rec.predict(gray[y:y+h,x:x+w])
	#	cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255);
	#cv2.imshow("Face",img);
	#if(cv2.waitKey(1)==ord('q')):
	#	break;
#cam.release()
#cv2.destroyAllWindows()