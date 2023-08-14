import cv2
import mediapipe as mp
import time

# Open first (probably front) camera
cap = cv2.VideoCapture(0)
pTime=0

# use mediapipe face detection sol
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

# initialize FD object
faceDetection = mpFaceDetection.FaceDetection(0.8)

while True:
  # Capture a frame every iteration
  success, img = cap.read()
  
  # convert to RGB image before sending to mediapipe
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = faceDetection.process(imgRGB)

  # check for multiple faces
  if results.detections:
    for id, detection in enumerate(results.detections):
      mpDraw.draw_detection(img, detection)
      # print(id, detection)
      # print(detection.location_data.relative_bounding_box)

      # store the bounding box from the class
      bboxC = detection.location_data.relative_bounding_box
      
      # get image h, w, channels
      ih, iw, ic = img.shape

      # our simple bounding box
      bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
              int(bboxC.width * iw), int(bboxC.height * ih)
      cv2.rectangle(img, bbox, (255, 0, 255), 2)

      # get confidence score
      cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), 
                  cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
      

  # display FPS
  cTime = time.time()
  fps = 1/(cTime-pTime)
  pTime=cTime
  cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
              3, (20, 255, 20), 3)
  
  # display image in a window
  cv2.imshow("Facecam", img)
  cv2.waitKey(1)