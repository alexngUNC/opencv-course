import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
  def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
    self.staticMode=staticMode
    self.maxFaces=maxFaces
    self.minDetectionCon=minDetectionCon
    self.minTrackCon=minTrackCon

    self.mpDraw = mp.solutions.drawing_utils
    self.mpFaceMesh = mp.solutions.face_mesh
    self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, max_num_faces=self.maxFaces,
                                             refine_landmarks=False, min_detection_confidence=self.minDetectionCon, 
                                             min_tracking_confidence=self.minTrackCon)
    self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

  def findFaceMesh(self, img, draw=True):
    self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.faceMesh.process(self.imgRGB)
    faces = []
    # draw face mesh if detected
    if self.results.multi_face_landmarks:
      for faceLms in self.results.multi_face_landmarks:
        if draw:
          self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
        face = []
        for id, lm in enumerate(faceLms.landmark):
          # get pixel coordinates
          ih, iw, ic = img.shape
          x, y = int(lm.x*iw), int(lm.y*ih)
          # print landmark IDs
          cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                      1, (0, 255, 0), 1)
          face.append([x, y])
        faces.append(face)
    return img, faces

def main():
  cap = cv2.VideoCapture(0)
  pTime=0
  detector = FaceMeshDetector()
  while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img)
    if len(faces) > 0:
      # print number of faces
      print(len(faces))
      # print points in first face
      # print(faces[0])
    # -- display FPS --
    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime
    cv2.putText(img, f'FPS: {fps}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 9), 3)
    cv2.imshow("Facecam", img)
    cv2.waitKey(1)

if __name__ == '__main__':
  main()