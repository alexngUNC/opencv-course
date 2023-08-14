import cv2
import time
import PoseModule2 as pm

cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.PoseDetector()
while True:
  success, img = cap.read()
  img = detector.findPose(img, draw=True)
  lmList = detector.findPosition(img)
  if len(lmList) > 0:
    cv2.circle(img, (lmList[0][1], lmList[0][2]), 25, (0, 0, 255), cv2.FILLED)
  
  # Calculate FPS
  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime=cTime
  cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
  cv2.imshow('Image', img)
  cv2.waitKey(1)