
# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
#from openalpr_ocr import ocr

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

#load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["wapon1.names"])
#belsPath = os.path.sep.join(["coco1.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

print("Loading ...................")
net = cv2.dnn.readNetFromDarknet('yolov3_testing.cfg', 'yolov3_training_2000.weights')
#net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
ln = net.getLayerNames()
ln = [ln[i- 1] for i in net.getUnconnectedOutLayers()]

##vs = cv2.VideoCapture('C:\\Users\\My Lappy\\Desktop\\Object Detection\\videos\\airport.mp4')
##vs = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
sample=0;
error=0
writer = None
(W, H) = (None, None)

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

while True:

        ret, frame = cap.read()
        if W is None or H is None:
                (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > args["confidence"]:
                                # scale the bounding box coordinates back relative to
                                # the size of the image, keeping in mind that YOLO
                                # actually returns the center (x, y)-coordinates of
                                # the bounding box followed by the boxes' width and
                                # height
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top
                                # and and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates,
                                # confidences, and class IDs
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                confidences[i])
        ##                        print(text)
                        text1="{}".format(LABELS[classIDs[i]])
                        print(text1)
##                        if(text1=="Helmet"):
##                                print('helmet detected')
####                                # ensure at least one detection exists
####                                if len(idxs) > 0:
####                                        # loop over the indexes we are keeping
####                                        for i in idxs.flatten():
####                                                # extract the bounding box coordinates
####                                                (x, y) = (boxes[i][0], boxes[i][1])
####                                                (w, h) = (boxes[i][2], boxes[i][3])
####
####                                                # draw a bounding box rectangle and label on the frame
####                                                color = [int(c) for c in COLORS[classIDs[i]]]
####                                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
####                                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
####                                                        confidences[i])
####                                ##                        print(text)
####                                                text1="{}".format(LABELS[classIDs[i]])
####                                                if(text1=="Helmet"):
####                                                        print("helmet")
####                                                        
####                        
####
####                                                else:
####                                                        print('NO HELMET FOUND')
####                                                        cv2.imwrite('frame.png',frame)
######                                                        print(ocr('frame.png'))

        cv2.imshow('name',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
        ##        Object.close()
                break

# release the file pointers
#writer.release()
cap.release()
