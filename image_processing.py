import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import torch # Import PyTorch library
import model # Import the model.py file

# load model.pt
model = model.Net()
model.load_state_dict(torch.load('model.pt'))
model.eval()

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

# Find the biggest contour
biggest_contour = np.array([])
max_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 50:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest_contour = approx
            max_area = area

# Reorder the points in the contour to go clockwise
def reorder(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new

# Warp the image
if biggest_contour.size != 0:
    biggest_contour = reorder(biggest_contour)
    cv2.drawContours(img_contours, biggest_contour, -1, (0, 0, 255), 10)
    pts1 = np.float32(biggest_contour)
    pts2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped = cv2.warpPerspective(img, matrix, (img_width, img_height))
    img_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

# Create a Sudoku grid
def split_grid(img):
    rows = np.vsplit(img, 9)
    squares = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            squares.append(box)
    return squares

squares = split_grid(img_warped)


# Predict the digits
def predict_digits(squares):
    result = []
    for image in squares:
        img = image[4:img_width-4, 4:img_height-4]
        img = cv2.resize(img, (28, 28))
        img = cv2.bitwise_not(img)
        img = img / 255.0
        img = torch.Tensor(img).view(-1, 1, 28, 28)
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        probability = torch.max(outputs.data, 1)[0].item()
        if probability > 0.7:
            result.append(predicted.item())
        else:
            result.append(0)
    return result



result = predict_digits(squares)
print(result)

# Display the image
cv2.imshow('Image', img_thresh)
cv2.imshow('Contours', img_contours)
cv2.imshow('Warped Image', img_warped)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
