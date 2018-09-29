import cv2
import numpy as np

# Video Capture 
capture = cv2.VideoCapture("demo.mov")

# Subtractors
mogSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG(300)
mog2Subtractor = cv2.createBackgroundSubtractorMOG2(300, 400, True)
gmgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG(10, .8)
knnSubtractor = cv2.createBackgroundSubtractorKNN(100, 400, True)
cntSubtractor = cv2.bgsegm.createBackgroundSubtractorCNT(5, True)

# Keeps track of what frame we're on
frameCount = 0

# Determine how many pixels do you want to detect to be considered "movement"
movementCount = 1000
movementText = "Someones stealing your honey"
textColor = (255, 255, 255)

while(1):
	# Return Value and the current frame
	ret, frame = capture.read()

	#  Check if a current frame actually exist
	if not ret:
		break

	frameCount += 1
	# Resize the frame
	resizedFrame = cv2.resize(frame, (0, 0), fx=0.375, fy=0.25)

	# Get the foreground masks using all of the subtractors
	mogMask = mogSubtractor.apply(resizedFrame)
	mog2Mmask = mog2Subtractor.apply(resizedFrame)
	gmgMask = gmgSubtractor.apply(resizedFrame)
	gmgMask = cv2.morphologyEx(gmgMask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
	knnMask = knnSubtractor.apply(resizedFrame)
	cntMask = cntSubtractor.apply(resizedFrame)

	# Count all the non zero pixels within the masks
	mogCount = np.count_nonzero(mogMask)
	mog2MCount = np.count_nonzero(mog2Mmask)
	gmgCount = np.count_nonzero(gmgMask)
	knnCount = np.count_nonzero(knnMask)
	cntCount = np.count_nonzero(cntMask)

	print('mog Frame: %d, Pixel Count: %d' % (frameCount, mogCount))
	print('mog2M Frame: %d, Pixel Count: %d' % (frameCount, mog2MCount))
	print('gmg Frame: %d, Pixel Count: %d' % (frameCount, gmgCount))
	print('knn Frame: %d, Pixel Count: %d' % (frameCount, knnCount))
	print('cnt Frame: %d, Pixel Count: %d' % (frameCount, cntCount))

	titleTextPosition = (100, 40)
	titleTextSize = 1.2
	cv2.putText(mogMask, 'MOG', titleTextPosition, cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
	cv2.putText(mog2Mmask, 'MOG2', titleTextPosition, cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
	cv2.putText(gmgMask, 'GMG', titleTextPosition, cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
	cv2.putText(knnMask, 'KNN', titleTextPosition, cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)
	cv2.putText(cntMask, 'CNT', titleTextPosition, cv2.FONT_HERSHEY_SIMPLEX, titleTextSize, textColor, 2, cv2.LINE_AA)

	stealingTextPosition = (100, 100)
	if (frameCount > 1):
		if (mogCount > movementCount):
			print(movementText)
			cv2.putText(mogMask, 'Someones stealing your honey', stealingTextPosition, cv2.FONT_HERSHEY_SIMPLEX, .5, textColor, 2, cv2.LINE_AA)
		if (mog2MCount > movementCount):
			print(movementText)
			cv2.putText(mog2Mmask, 'Someones stealing your honey', stealingTextPosition, cv2.FONT_HERSHEY_SIMPLEX, .5, textColor, 2, cv2.LINE_AA)
		if (gmgCount > movementCount):
			print(movementText)
			cv2.putText(gmgMask, 'Someones stealing your honey', stealingTextPosition, cv2.FONT_HERSHEY_SIMPLEX, .5, textColor, 2, cv2.LINE_AA)
		if (knnCount > movementCount):
			print(movementText)
			cv2.putText(knnMask, 'Someones stealing your honey', stealingTextPosition, cv2.FONT_HERSHEY_SIMPLEX, .5, textColor, 2, cv2.LINE_AA)
		if (cntCount > movementCount):
			print(movementText)
			cv2.putText(cntMask, 'Someones stealing your honey', stealingTextPosition, cv2.FONT_HERSHEY_SIMPLEX, .5, textColor, 2, cv2.LINE_AA)

	cv2.imshow('Original', resizedFrame)
	cv2.imshow('MOG', mogMask)
	cv2.imshow('MOG2', mog2Mmask)
	cv2.imshow('GMG', gmgMask)
	cv2.imshow('KNN', knnMask)
	cv2.imshow('CNT', cntMask)
	
	cv2.moveWindow('Original', 0, 0)
	cv2.moveWindow('MOG', 0, 315)
	cv2.moveWindow('KNN',  0, 605)
	cv2.moveWindow('GMG', 719, 0)
	cv2.moveWindow('MOG2', 719, 315)
	cv2.moveWindow('CNT', 719, 605)

	k = cv2.waitKey(0) & 0xff
	if k == 27:
		break

capture.release()
cv2.destroyAllWindows()
