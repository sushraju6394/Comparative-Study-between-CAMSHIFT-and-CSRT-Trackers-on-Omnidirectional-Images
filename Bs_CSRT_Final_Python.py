from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import numpy as np

filename='G:/Video3_Full/Record_%05d.png'
cap=cv2.VideoCapture(filename)

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    }
trackerName="csrt"     
# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()
fgbg2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
# if a video path was not supplied, grab the reference to the web cam

correct=0
wrong=0
a=[]
efficiency=0
count=0
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    #cap.set(2,frame_no)
    ret,frame = cap.read()
    #print (len(frame))
    count=count+1
    print('frame number',count)
    
    #print('dimentions : ',frame.shape)
    BS_frame=frame
    #BS_frame = imutils.resize(BS_frame, width=600)
 
    # check to see if we have reached the end of the stream

    # resize the frame (so we can process it faster)
    #frame = imutils.resize(frame, width=600)

    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    #global x,y,w,h,rectangle
    (success, boxes) = trackers.update(frame)
 
    # loop over the bounding boxes and draw then on the frame
    for box in boxes:
        
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  

    # show the output frame
    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 1680,1680)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)
              
        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
        tracker = OPENCV_OBJECT_TRACKERS[trackerName]()
        trackers.add(tracker, frame, box)
    dst = cv2.GaussianBlur(BS_frame,(5,5),cv2.BORDER_DEFAULT)
    fgmask2=fgbg2.apply(BS_frame)
    #frame = imutils.resize(fgmask2, width=600)
    #fgmask2= cv2.resize(fgmask2,(1200,1600))
    #print(type(fgmask2))
       
    for box in boxes:
        
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(fgmask2, (x, y), (x + w, y + h), (255), 2)
        roi=fgmask2[y:y+h,x:x+w]
        a = cv2.countNonZero(roi)
        Percentage=(a/(w*h)*100)
        print('percentage of white pixels is',Percentage)
        #print(a)
        if(Percentage<30):
            wrong=wrong+1
            #print (wrong)
            print('tracking not done')
        else:
            correct=correct+1
            #print (correct)
            print('tracking successful')
    
        cv2.namedWindow('MOG2',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MOG2`', 1680,1680)
        cv2.imshow('MOG2', fgmask2)
        efficiency=(correct/(correct+wrong))*100
        
  # if the `q` key was pressed, break from the loop
    if key == ord("q"):
           break    
print('efficiency of csrt is', efficiency)
print('the number of tracked frames is',correct)
print('the number of non tracked frames is',wrong)

# close all windows
cv2.destroyAllWindows()
