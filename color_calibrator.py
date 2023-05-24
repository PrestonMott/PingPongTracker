import cv2

# Video file path
video_path = 'path/to/video/file.mp4'

# Function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Retrieve the BGR values of the pixel
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[y, x]
        # Display RGB values
        print(f"HSV: {h}, {s}, {v}")

# Create a window for displaying the video
cv2.namedWindow('Color Calibrator')
# Set mouse callback function for the window
cv2.setMouseCallback('Color Calibrator', mouse_callback)

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame = cap.read()

while ret:
    # Display the frame
    cv2.imshow('Color Calibrator', frame)

    # Check for keyboard events
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read the next frame
    ret, frame = cap.read()

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()