import cv2 as cv

img = cv.imread('green_fish.jpg')

cv.imshow('Fish', img)

cv.waitKey(0)