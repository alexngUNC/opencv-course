import cv2
import mediapipe as mp
import time

class handDetector():
  def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
    self.mode = mode
    self.maxHands = maxHands
    self.detectionCon = detectionCon
    self.trackCon = trackCon
    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detectionCon, self.trackCon)
    self.mpDraw = mp.solutions.drawing_utils


  def findHands(self, img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)

    if self.results.multi_hand_landmarks:
      for handLm in self.results.multi_hand_landmarks:
        if draw:
          self.mpDraw.draw_landmarks(img, handLm, self.mpHands.HAND_CONNECTIONS)
    return img
  

  def findPosition(self, img, handNo=0, draw=True):
    lmList = []
    if self.results.multi_hand_landmarks:
      myHand = self.results.multi_hand_landmarks[handNo]
      for id, lm in enumerate(myHand.landmark):
        h,w,c = img.shape
        cx, cy = int(lm.x*w), int(lm.y*h)
        lmList.append([id, cx, cy])
        if draw:
          cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    return lmList




# while True:
#   success, img = cap.read()
#   imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#   results = hands.process(imgRGB)

#   #print(results.multi_hand_landmarks)

#   # check if we have multiple hands
#   if results.multi_hand_landmarks:
#     for handLms in results.multi_hand_landmarks:
#       for id, lm in enumerate(handLms.landmark):
#         # landmark is a finger
#         # get height width and channels of image
#         h, w, c = img.shape

#         # find position of finger
#         cx, cy = int(lm.x*w), int(lm.y*h)

#         # draw circle on thumb
#         if id == 4:
#           cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
#       mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

#   # update time
#   cTime = time.time()
#   fps = 1 / (cTime - pTime)
#   pTime = cTime

#   # display time on image
#   cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
#               (255, 0, 255), 3)

#   cv2.imshow("Image", img)
#   cv2.waitKey(1)


def main():
  # previous time
  pTime = 0

  # current time
  cTime = 0

  # get video capture
  cap = cv2.VideoCapture(0)

  detector = handDetector()
  while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
      # print thumb position
      print(lmList[4])
    # update time
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # display time on image
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == '__main__':
  main()