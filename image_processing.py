from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitRecogNetwork(nn.Module):
    def __init__(self, num_classes):
        super(DigitRecogNetwork, self).__init__()
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
model = DigitRecogNetwork(10)
model.load_state_dict(torch.load('model.pth'))
model.eval()

def find_puzzle(img):
    # image preprocessing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # find contours in thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

    # find largest quadrilateral contour
    puzzle_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri*0.02, True)

        if len(approx) == 4:
            puzzle_cnt = approx
            break

    # if puzzle contour found
    if puzzle_cnt is not None:
    
        # for testing purposes
        # output = img.copy()
        # cv2.drawContours(output, [puzzle_cnt], -1, (0,255,0), 2)
        # cv2.imshow('Puzzle Outline', output)
        
        # transform image to get top-down view
        puzzle_warped_gray = four_point_transform(img_gray, puzzle_cnt.reshape(4,2))

        # for testing purposes
        # cv2.imshow('Puzzle Transform', puzzle_warped_gray)

    return puzzle_warped_gray

def predict(cell):
    # resize and preprocess cell
    cell = cv2.resize(cell, (28,28))
    cell = cell.reshape(1, 1, 28, 28)
    cell = torch.from_numpy(cell)
    cell = cell.to(torch.float32)
    # predict digit
    prediction = model(cell)
    prediction = prediction.argmax(dim=1, keepdim=True)
    return prediction

def extract_digit(cell):
    # threshold and remove border
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    # find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # if no contours found
    if len(cnts) == 0:
        return None
    
    # find largest contour
    largest_cnt = max(cnts, key = cv2.contourArea)
    largest_cnt_img = np.zeros(thresh.shape, dtype = 'uint8')
    cv2.drawContours(largest_cnt_img, [largest_cnt], -1, 255, -1)

    # check if cell is empty
    (h,w) = thresh.shape
    percentFilled = cv2.countNonZero(largest_cnt_img)/float(w*h)
    if percentFilled < 0.03:
        return None
    
    return largest_cnt_img



def predict_all(img):
    warped = find_puzzle(img)
    # create empty board
    board = np.zeros((9,9), dtype = 'int')
    # calculate x and y step sizes
    x_step = warped.shape[1] // 9
    y_step = warped.shape[0] // 9


    for y in range(0,9):
        for x in range(0,9):
            # calculate cell coordinates
            x_start = x * x_step
            y_start = y * y_step
            x_end = (x+1) * x_step
            y_end = (y+1) * y_step

            # extract and predict digit and update board
            cell = warped[y_start:y_end, x_start:x_end]
            digit = extract_digit(cell)
            if digit is not None:
                digit = predict(digit)
                board[y,x] = digit
    return board