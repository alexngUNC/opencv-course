import cv2
import mediapipe as mp
import time

class FaceDetector():
  def __init__(self, minDetectionCon=0.5):
    self.minDetectionCon = minDetectionCon
    # use mediapipe face detection sol
    self.mpFaceDetection = mp.solutions.face_detection
    self.mpDraw = mp.solutions.drawing_utils

    # initialize FD object
    self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

  def findFaces(self, img, draw=True):
    # convert to RGB image before sending to mediapipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.faceDetection.process(imgRGB)

    # list of bounding boxes
    bboxs = []

    # check for face(s)
    if self.results.detections:
      for id, detection in enumerate(self.results.detections):
        # store the bounding box from the class
        bboxC = detection.location_data.relative_bounding_box
        
        # get image h, w, channels
        ih, iw, ic = img.shape

        # our simple bounding box
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                int(bboxC.width * iw), int(bboxC.height * ih)
        bboxs.append([id, bbox, detection.score])

        if draw:
          # draw the bounding box
          img = self.fancyDraw(img, bbox)

          # display confidence score
          cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), 
                      cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return img, bboxs

  def fancyDraw(self, img, bbox, l=30, t=5):
    x, y, w, h = bbox
    x1, y1 = x+w, y+h
    
    # draw lines at top left corner (x, y)
    cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
    cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)

    # draw top right corner (x1, y)
    cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)
    cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)

    # draw bottom right corner (x1, y1)
    cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
    cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 255), t)

    # draw bottom left corner (x, y1)
    cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)
    cv2.line(img, (x, y1), (x, y1-l), (255, 0, 255), t)

    # draw the bounding box
    cv2.rectangle(img, bbox, (255, 0, 255), 1)

    return img

def main():
  # prev time
  pTime=0

  # Open first (probably front) camera
  cap = cv2.VideoCapture(0)
  detector = FaceDetector(0.8)
  while True:
    # Capture a frame every iteration
    success, img = cap.read()

    # find faces
    img, bboxs = detector.findFaces(img)
    if len(bboxs) > 0:
      print(bboxs)
    # -- display FPS --
    # current time
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (20, 255, 20), 3)
    
    # display image in a window
    cv2.imshow("Facecam", img)
    cv2.waitKey(1)

if __name__ == '__main__':
  main()