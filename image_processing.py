import cv2 # Import the OpenCV library

img_path = 'sudoku_test.jpg'
img_height = 450
img_width = 450

# Read the image
img = cv2.imread(img_path)
img = cv2.resize(img, (img_width, img_height))

#Preprocess the image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
img_thresh = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)

# Find the contours
img_contours = img.copy()
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

# Display the image
cv2.imshow('Image', img_thresh)
cv2.imshow('Contours', img_contours)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
