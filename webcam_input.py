import cv2 # Import the OpenCV library

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # Read a frame from the webcam

    cv2.imshow('Webcam', frame) # Display the frame

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # Release the webcam
cv2.destroyAllWindows() # Close all windows
