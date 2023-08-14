import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

#########################
# Camera width and height
wCam, hCam = 640, 480
#########################
cap = cv2.VideoCapture(1)

# prop ID 3 is width cam
cap.set(3, wCam)
# prop ID 4 is height cam
cap.set(4, hCam)
pTime = 0
volBar, volPer = 0, 0
# create detector object
detector = htm.handDetector(detectionCon=0.8)
while True:
  # get a frame from the facecam
  success, img = cap.read()

  # find hands
  img = detector.findHands(img)

  # get hand coordinates
  lmList = detector.findPosition(img, draw=False)
  
  # make sure there are some points
  if len(lmList) > 0:  
    # tip of thumb and index finger, respectively
    x1, y1 = lmList[4][1], lmList[4][2]
    x2, y2 = lmList[8][1], lmList[8][2]
    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
    cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

    # draw a line between them
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    # get center of line
    cx, cy = (x1+x2)//2, (y1+y2)//2
    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    # get length of line
    length = math.hypot(x2-x1, y2-y1)
    # print(length)

    # convert length to volumne
    # np.interp(length, [50, 300], [minVol, maxVol])
    # set the volume
    # setVolume(vol  )
    # make vol bar
    volBar = np.interp(length, [50, 300], [400, 150])
    volPer = np.interp(length, [50, 300], [0, 100])

    # change color of center circle like a button
    if length < 50:
      cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
  # display volume bar
  cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
  cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
  
  # display volume %
  cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN,
              1, (200, 80, 10), 3)
  # calc fps
  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime = cTime

  # display fps
  cv2.putText(img, f'FPS: {int(fps)}', (50, 70), cv2.FONT_HERSHEY_PLAIN,
              1, (0, 255, 0), 2)
  
  # display img
  cv2.imshow("Facecam", img)
  cv2.waitKey(1)