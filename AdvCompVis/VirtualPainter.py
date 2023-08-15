import cv2
import numpy as np
import os
import time
import HandTrackingModule as htm

brushThickness = 15
eraserThickness = 100

# folderPath = 'Header'
# myList = os.listdir(folderPath)

# image canvas
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# overlays
blankBlack = np.zeros((125, 1280, 3), np.uint8)
overlayList = [blankBlack, blankBlack, blankBlack, blankBlack, blankBlack]
# for imPath in myList:
#   image = cv2.imread(f'{folderPath}/{imPath}')
#   overlayList.append(image)

# header image
header = overlayList[0]

# set default draw color
drawColor = (255, 0, 255)

# run webcam
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# hand detector
detector = htm.handDetector(detectionCon=0.85)

# x and y previous
xp, yp = 0, 0

while True:
  # get frame from facecam
  success, img = cap.read()

  # flip image horizontally
  img = cv2.flip(img, 1)

  # find hand landmarks
  img = detector.findHands(img)
  lmList = detector.findPosition(img, draw=False)

  # if lms are detected
  if len(lmList) > 0:
    # tip of index and middle fingers coordinates
    x1, y1 = lmList[8][1:]
    x2, y2 = lmList[12][1:]

    # check which fingers are up
    fingers = detector.fingersUp()
    # print(fingers)

    # if selection mode - two fingers are up
    if fingers[1] and fingers[2]:
      xp, yp = 0, 0
      cv2.rectangle(img, (x1, y1-25), (x2, y2+25), (255, 0, 255), cv2.FILLED)
      print('SELECTION MODE')

      # check if in the hedaer
      if y1 < 125:
        if 250 < x1 < 450:
          header = overlayList[0]
          drawColor = (255, 0, 0)
        elif 550 < x1 < 750:
          header = overlayList[1]
          drawColor = (255, 0, 0)
        elif 800 < x1 < 950:
          header = overlayList[2]
          drawColor = (0, 255, 0)
        elif 1050 < x1 < 1200:
          header = overlayList[3]
          drawColor = (0, 0, 0)
        cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
    
    # if drawing mode - index finger is up
    if fingers[1] and fingers[2] == False:
      cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
      print('DRAWING MODE')
      if xp == 0 and yp == 0:
        xp, yp = x1, y1

      # check if erasing
      if drawColor == (0, 0, 0):
        cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
      # draw a line
      else:
        cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
      xp, yp = x1, y1
  
  # gray image for drawing on facecam
  imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
  _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
  imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
  img = cv2.bitwise_and(img, imgInv)
  img = cv2.bitwise_or(img, imgCanvas)
  
  # overlay the header
  img[0:125, 0:1280] = header
  
  # blend images
  # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

  cv2.imshow('Facecam', img)
  # cv2.imshow('Canvas', imgCanvas)
  # cv2.imshow('Inv', imgInv)
  cv2.waitKey(1)
