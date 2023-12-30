import image_processing
import cv2

img_path = 'sudoku_test.jpg'

# NOTE : this doesnt work well because thresholds image not good before prediction.

img_processed = image_processing.process_image(img_path)
contours = image_processing.contours(img_processed)
biggest_contour = image_processing.biggest_contour(contours)
biggest_contour = image_processing.reorder(biggest_contour)

img = cv2.imread(img_path)
img_resize = cv2.resize(img, (450, 450))

#Preprocess the image
img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)

img_warped = image_processing.warp(img_blur, biggest_contour)
squares = image_processing.split_grid(img_warped)
predicted = image_processing.predict_digits(squares)
print(predicted)

cv2.imshow('Image', img_processed)
cv2.imshow('Contours', img_warped) 
# show 8th box
cv2.imshow('8th box', squares[7])
cv2.waitKey(0)

