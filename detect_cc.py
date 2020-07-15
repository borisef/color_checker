# import the necessary packages
import time
import cv2, os
import numpy as np
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
from cc_utils import resizeTo, GetRGBs, GetTemplate, FindRGBTransform, FindRGBTransformPLS, ApplyRGBTransform, ApplyRGBTransformPLS, FindRGBTransformWLS

NUM_COLORS = 24


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
		# draw a rectangle around the region of interest
		cv2.rectangle(imsmall, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("zoom2cc", imsmall)

def click_and_keep(event, x, y, flags, param):
	# grab references to the global variables
	global refPt
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.append([x, y])
		L = len(refPt)
		print(L)
		cv2.circle(roi,(x,y),10,bgrTemplate[L-1,:], -1)
		cv2.imshow("ROI", roi)



# construct the argument parser and parse the arguments
im = "cc/00001565.tif"
im = "cc/cm_20200709_160029_XX_tif/00001177.tif"
#im = "template/cc_sun.jpg"

folder2process = "ccs"
outFolder = "new_ccs"

#im = "cc/00001640.tif"
#im = "cc/00001479.tif"
# load the image, clone it, and setup the mouse callback function

image = cv2.imread(im) #BGR
bgrTemplate = GetTemplate() #BGR
bgrTemplate = bgrTemplate[0:NUM_COLORS,:]


lenPtr = 0
lastLenPtr = 0

imsmall, rsz = resizeTo(image,1000)
clone_small = imsmall.copy()

cv2.namedWindow("zoom2cc")
cv2.setMouseCallback("zoom2cc", click_and_crop)

#zoom to correct region
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("zoom2cc", imsmall)
	key = cv2.waitKey(1) & 0xFF
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		imsmall = clone_small.copy()
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
	cv2.destroyWindow("zoom2cc")
	refPt = (np.array(refPt)/rsz).astype(int)
	roi = image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	roi, rsz = resizeTo(roi,500)

	cv2.imshow("ROI", roi)
	cv2.waitKey(delay = 1)
# close all open windows
cv2.destroyAllWindows()


refPt = []

cv2.namedWindow("ROI")
cv2.setMouseCallback("ROI", click_and_keep)
clone_roi = roi.copy()
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("ROI", roi)
	key = cv2.waitKey(1) & 0xFF
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		roi = clone_roi.copy()
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
	if(len(refPt) != lastLenPtr):
		lastLenPtr = len(refPt)
		print(len(refPt))
	if len(refPt) == NUM_COLORS:

		break

cv2.destroyWindow("ROI")

cv2.namedWindow("inROI")
cv2.imshow("inROI", roi)
# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 4:
	#TODO: generate NUM_COLORS points and show
	for i in range(4):
		cv2.rectangle(roi, (refPt[i][0]-3,refPt[i][1]-3), (refPt[i][0]+3,refPt[i][1]+3 ), (0, 255, 0), 2)
	cv2.imshow("inROI", roi)
	cv2.waitKey(0)

# if len(refPt) == NUM_COLORS:
# 	# generate NUM_COLORS points and show
# 	for i in range(NUM_COLORS):
# 		cv2.circle(roi, (refPt[i][0], refPt[i][1]), 10, bgrTemplate[i, :], -1)
#
# 	print(refPt)
# 	cv2.imshow("inROI", roi)
# 	cv2.waitKey(0)
# # close all open windows
cv2.destroyAllWindows()



#select NUM_COLORS locations

#pick colors
RGB24 = GetRGBs(clone_roi,refPt)
print("RGBs")
print(RGB24)
RGB24 = np.array(RGB24)
BGR24 = RGB24[:,::-1] #BGR again
#find transform
T = FindRGBTransform(rgbFrom=BGR24, rgbTo= bgrTemplate)
#T= FindRGBTransformWLS(rgbFrom=RGB24, rgbTo= bgrTemplate)
#tranform orig image
newIm = ApplyRGBTransform(image, T)

newIm = cv2.convertScaleAbs(newIm)

cv2.namedWindow("NEW")
cv2.imshow("NEW", resizeTo(newIm,1000)[0])
cv2.namedWindow("OLD")
cv2.imshow("OLD", resizeTo(image,1000)[0])

cv2.imwrite("res.png",newIm)

cv2.waitKey(0)


if(folder2process is not None):
	if(not os.path.exists(outFolder)):
		os.mkdir(outFolder)
	for f in os.listdir(folder2process):
		im = cv2.imread(os.path.join(folder2process,f))
		newIm = ApplyRGBTransform(im, T)
		newIm = cv2.convertScaleAbs(newIm)
		newName = os.path.join(outFolder,f)
		cv2.imwrite(newName, newIm)




print("OK")

#see also
#https://blog.francium.tech/using-machine-learning-for-color-calibration-with-a-color-checker-d9f0895eafdb
