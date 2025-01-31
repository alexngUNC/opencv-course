import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime=0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
while True:
  success, img = cap.read()
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = faceMesh.process(imgRGB)

  # draw face mesh if detected
  if results.multi_face_landmarks:
    for faceLms in results.multi_face_landmarks:
      mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpecs, drawSpecs)

      for lm in faceLms.landmark:
        # print each landmark
        #print(lm)

        # get pixel coordinates
        ih, iw, ic = img.shape
        x, y = int(lm.x*iw), int(lm.y*ih)
        print(id, x, y)
  # -- display FPS --
  cTime = time.time()
  fps = int(1/(cTime-pTime))
  pTime = cTime
  cv2.putText(img, f'FPS: {fps}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
              3, (0, 255, 9), 3)
  cv2.imshow("Facecam", img)
  cv2.waitKey(1)