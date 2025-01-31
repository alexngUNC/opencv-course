import cv2 as cv

# img = cv.imread('Photos/cat_large2.jpg')
# cv.imshow('Cat', img)

def rescaleFrame(frame, scale=0.75):
  # Works for images, videos, and live video
  width = int(frame.shape[1] * scale)
  height = int(frame.shape[0] * scale)
  dimensions = (width, height)

  return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
  # Only works for live video
  capture.set(3, width)
  capture.set(4, height)


# Reading videos
capture = cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
  isTrue, frame = capture.read()

  frame_resized = rescaleFrame(frame)
  
  cv.imshow('Video', frame)
  cv.imshow('Video Resized', frame_resized)

  # If the d key is pressed, stop
  if cv.waitKey(20) & 0xFF==ord('d'):
    break
   
capture.release()
cv.destroyAllWindows()