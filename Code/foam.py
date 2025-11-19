import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture('Beer_05.avi')

# Define a function to find foam height in a frame
def get_foam_height(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to isolate foam (adjust the threshold as needed)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of the foam (white regions in the thresholded image)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming foam is the largest contour, find its bounding box
    if contours:
        foam_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(foam_contour)  # Bounding box coordinates
        
        # Return the height of the foam
        return h
    return 0

# Loop over video frames and calculate foam height
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    foam_height = get_foam_height(frame)
    
    # Display the foam height on the frame (optional)
    cv2.putText(frame, f"Foam Height: {foam_height} px", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame (optional)
    cv2.imshow('Frame', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
