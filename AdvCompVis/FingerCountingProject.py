import cv2
import time
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(1)
# set cam width and height
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(3, hCam)

# get list of images
# folderPath = 'FingerImages'
# myList = os.listdir(folderPath)
# overlayList = []
# for imPath in myList:
#   image = cv2.imread(f'{folderPath}/{imPath}')
#   overlayList.append(image)
pTime = 0

# create hand tracker
detector = htm.handDetector(detectionCon=0.8)

# finger tip IDs:
# thumb, index, middle, ring, pinky
tipIds = [4, 8, 12, 16, 20]

while True:
  # read a frame from the camera
  success, img = cap.read()

  # find hands
  img = detector.findHands(img)

  # landmark list
  lmList = detector.findPosition(img, draw=False)
  
  # determine if fingers are open or closed based on 
  # tips of fingers being below the i
  fingers = []
  if (len(lmList) > 0):
    # check if thumb is open (to the right); right-hand dependent
    if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
      fingers.append(1)
    else:
      fingers.append(0)

    # check if finger is up (y coordinate)
    for id in range(1, 5):
      # if the finger is open, append 1
      if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]: # opencv coords are top-down
        fingers.append(1)
      else:
        fingers.append(0)
  # print(fingers)
  totalFingers=fingers.count(1)
  print(totalFingers)
    

  # overlay the image in a box in the lop left
  # curImg = overlayList[totalFingers-1]
  # h, w, c = curImg.shape
  # img[0:h, 0:w] = curImg

  # display count
  cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
  cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
              10, (200, 100, 20), 25)

  # display FPS
  cTime = time.time()
  fps = int(1/(cTime-pTime))
  pTime = cTime
  cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
              3, (100, 0, 255), 3)

  cv2.imshow("Facecam", img)
  cv2.waitKey(1)