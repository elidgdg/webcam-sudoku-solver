import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import torch # Import PyTorch library
# from model import LargerModel
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LargerModel(nn.Module):
    def __init__(self, num_classes):
        super(LargerModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 15, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

# load model.pt
model = LargerModel(10)
model.load_state_dict(torch.load('model.pth'))
model.eval()

img_path = 'sudoku_test.jpg'

def process_image(img_path, img_width=450, img_height=450):
    # Read the image
    # img = cv2.imread(img_path)
    img_resize = cv2.resize(img_path, (img_width, img_height))

    #Preprocess the image
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)

    return img_thresh

def get_contours(img):   
    # Find the contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img_contours = img.copy()
    # cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
    return contours

def get_biggest_contour(contours): 
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
    return biggest_contour

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
def warp(img, biggest_contour, img_width=450, img_height=450):
    pts1 = np.float32(biggest_contour)
    pts2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped = cv2.warpPerspective(img, matrix, (img_width, img_height))
    return img_warped

# if biggest_contour.size != 0:
#     biggest_contour = reorder(biggest_contour)
#     cv2.drawContours(img_contours, biggest_contour, -1, (0, 0, 255), 10)
#     pts1 = np.float32(biggest_contour)
#     pts2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
#     matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     img_warped = cv2.warpPerspective(img, matrix, (img_width, img_height))
#     img_warped_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

# Create a Sudoku grid
def split_grid(img):
    rows = np.vsplit(img, 9)
    squares = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            squares.append(box)
    return squares

# squares = split_grid(img_warped)

# Predict the digits
def predict_digits(squares, img_width=450, img_height=450):
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
    
    # reshape result into 9x9 grid
    result = np.array(result).reshape((9, 9))
    return result

def predict_main(img):
    img_processed = process_image(img)
    contours = get_contours(img_processed)
    biggest_contour = get_biggest_contour(contours)
    biggest_contour = reorder(biggest_contour)
    img_warped = warp(img_processed, biggest_contour)
    squares = split_grid(img_warped)
    predicted = predict_digits(squares)
    return predicted

# result = predict_digits(squares)
# print(result)

# # Display the image
# cv2.imshow('Image', img_thresh)
# cv2.imshow('Contours', img_contours)
# cv2.imshow('Warped Image', img_warped)

# # Wait for a key press to exit
# cv2.waitKey(0)
# cv2.destroyAllWindows()
