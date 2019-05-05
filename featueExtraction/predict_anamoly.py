
from __future__ import print_function
import cv2
import numpy as np
import pickle
import os
import scipy.misc


DSDir="/home/ichigo/Desktop/AnomalyProject/DataSet/VideoSet/" 	#DataSet Directory
videoFramesDir = os.path.dirname(os.path.abspath(__file__)) + "/frames/"

clfNames = ["NearestNeighbors", "LinearSVM", "RBFSVM", "DecisionTree",
		 "RandomForest", "AdaBoost", "QuadraticDiscriminantAnalysis", "LinearDiscriminantAnalysis", "NaiveBayes", "Neural Networks", "Decision Trees"]

def draw_flow(img, flow, step=16):
	h, w = img.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
	fx, fy = flow[y,x].T
	lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.5)
	vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cv2.polylines(vis, lines, 0, (0, 255, 0))
	for (x1, y1), (x2, y2) in lines:
		cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
	return vis


def process_atom(bins, magnitude, fmask, classifier, frameCount, inputVideoFrame, atom_shape=[10,10,5]):

	bin_count = np.zeros(9, np.uint8)
	h,w, t = bins.shape
	
	tagged = np.zeros((h, w, 1), np.uint8)
	for i in range(0,h,atom_shape[0]):
		for j in range(0, w, atom_shape[1]):
			i_end = min(h, i+10)
			j_end = min(w, j+10)
			# Get the atom for bins
			atom_bins = bins[i:i_end, j:j_end].flatten()

			# Average magnitude
			atom_mag = magnitude[i:i_end, j:j_end].flatten().mean()
			atom_fmask = fmask[i:i_end, j:j_end].flatten()
			# Count of foreground values
			f_cnt = np.count_nonzero(atom_fmask)

			# Get the direction bins values
			hs, _ = np.histogram(atom_bins, np.arange(10))

			features = hs.tolist()
			features.extend([f_cnt, atom_mag])
			features=np.array(features).reshape(1, -1)
			tag = classifier.predict(features)[0]
			tagged[i:i_end, j:j_end] = tag * 255

	cv2.imwrite(videoFramesDir + "taggedFrames/frame%d.jpg" % frameCount, tagged) # Save Tagged npArray To jpg Frames
	cv2.imwrite(videoFramesDir + "inputVideoFrames/frame%d.jpg" % frameCount, inputVideoFrame) # Save Input Video To jpg Frames
	
	convexHullBound(frameCount)
	return 0


def getPredictionForVideo(video_link, classifier, mag_threshold=1e-3, atom_shape=[10,10,5]):

	cam = cv2.VideoCapture(video_link)
	# first frame in grayscale
	ret, prev = cam.read()

	prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
	## Background extractor
	fgbg = cv2.createBackgroundSubtractorMOG2()
	h, w = prevgray.shape[:2]
	bins = np.zeros((h, w, atom_shape[2]), np.uint8)
	mag = np.zeros((h, w, atom_shape[2]), np.float32)
	fmask = np.zeros((h,w,atom_shape[2]), np.uint8)
	tag_img = np.zeros((h,w,atom_shape[2]), np.uint8)

	time = 0
	frameCount = 0
	# Go through all the frames of the video
	while True:
		#Read next frame
		ret, img = cam.read()
		#img = cv.imread(imagelist.__next__(), cv2.IMREAD_GRAYSCALE)
		if(ret==0):
			break
		# Get foreground/background
		fmask[...,time] = fgbg.apply(img)
		#Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# Calculate Optical Flow for all pixels in the image
		# Parameters :
		#       prevgray = prev frame
		#       gray     = current frame
		#       levels, winsize, iterations, poly_n, poly_sigma, flag
		#       0.5 - image pyramid or simple image scale
		#       3 - no of pyramid levels
		#       15 - window size
		#       3 - no of iterations
		#       5 - Polynomial degree epansion
		#       1.2 - standard deviation to smooth used derivatives
		#       0 - flag
		flow = cv2.calcOpticalFlowFarneback(prevgray, gray,None, 0.5, 3, 15, 3, 5, 1.2, 0)
		## Flow contains vx, vy for each pixel
		# Calculate direction and magnitude
		height, width = flow.shape[:2]
		fx, fy = flow[:,:,0], flow[:,:,1]

		# Calculate direction qunatized into 8 directions
		angle = ((np.arctan2(fy, fx+1) + 2*np.pi)*180)% 360
		binno = np.ceil(angle/45)

		# Calculate magnitude
		magnitude = np.sqrt(fx*fx+fy*fy)

		# Add to zero bin if magnitude below a certain threshold
		#if(magnitude < mag_threshold):
		binno[magnitude < mag_threshold] = 0

		bins[...,time] = binno
		mag[..., time] = magnitude
		time = time + 1

		if(time == 5):
			time = 0
			process_atom(bins, mag, fmask, classifier, frameCount, img)
			frameCount += 1

		prevgray = gray

	if(time > 0):
		process_atom(bins,mag,fmask, classifier, frameCount, img)
		frameCount += 1


def convexHullBound(frameCount): #TO PRINT BOUNDING BOX
	
	inputVideoFrame = cv2.imread(videoFramesDir + "inputVideoFrames/frame%d.jpg" % frameCount, 1)
	imshow('inputVideo', inputVideoFrame)

	outputFrame = inputVideoFrame

	taggedFrame = cv2.imread(videoFramesDir + "taggedFrames/frame%d.jpg" % frameCount, 1)

	gray = cv2.cvtColor(taggedFrame, cv2.COLOR_BGR2GRAY)
	blur = cv2.blur(gray, (3, 3))
	ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

	# Finding contours for the thresholded image
	_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# create hull array for convex hull points
	hull = []

	# calculate points for each contour
	for i in range(len(contours)):
		# creating convex hull object for each contour
		hull.append(cv2.convexHull(contours[i], False))

	# create an empty black image
	drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
	
	# draw contours and hull points
	for i in range(len(contours)):
		color_contours = (0, 255, 0) # green - color for contours
		color = (0, 0, 255) # blue - color for convex hull
		# draw ith contour
		cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
		# draw ith convex hull object
		cv2.drawContours(drawing, hull, i, color, 1, 8)
		cv2.drawContours(outputFrame, hull, i, color, 2, 8)

	imshow('outputVideo', outputFrame)
	

def imshow(windowName, npData):
	try:
		cv2.imshow(windowName, npData)
		cv2.waitKey(5)
	except:
		cv2.destroyAllWindows()


def main():

	DSType="Ped" + input("Input DataSet Type (Ped1/Ped2):")
	threshold=input("Enter classification threshold (10/50):")

	if(DSType=="Ped1"):
		print("\nVideo: 1 - 36")
		VidNo=input("Video Number:")
	if(DSType=="Ped2"):
		print("\nVideo: 1 - 12")
		VidNo=input("Video Number:")
	if(int(VidNo) < 10):
		VidNo = "00" + VidNo
	else:
		VidNo = "0" + VidNo

	# print classifier Names
	print("\nClassifiers :")
	count=1
	for i in clfNames:
		print(str(count) +'. '+ i)
		count += 1

	Choice = int(input("Enter the classifier Choice:"))-1
	print('\n' + clfNames[Choice])

	CLASSIFIER_DIR = "../ML-Model/TrainedClassifiers/" + threshold + "/" + DSType + "/"
	clf = pickle.load(open(CLASSIFIER_DIR + clfNames[Choice] + '.pkl', 'rb'))

	video = DSDir + DSType + "/Test/Test" + VidNo + ".avi"
	getPredictionForVideo(video, clf)


if __name__ == "__main__":main()