#!/usr/bin/env python3
"""
Track the ball
"""

from numba import jit, cuda
import cv2
import numpy as np
import imutils
from collections import deque
import time

# used to record the time when we processed last frame
prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0
 
def calculate_fps(frame):
    global prev_frame_time, new_frame_time
    # Time when we finish processing this frame
    new_frame_time = time.time()
  
    # Number of frames processed in given time
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # Converting the fps into integer to round
    fps = int(fps)

    # Convert fps to string to display
    fps = str(fps)

    # Putting the FPS count on the frame
    cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

def calculate_trail(pts, buffer, center, frame):
    # Update points in queue
    pts.appendleft(center)

    for i in range(1, len(pts)):
        # If either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # Compute the thickness of the line and
        thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
        
        # Draw the connecting lines
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

def ball_tracker(frame):
    # Define the color range of the ping pong ball in HSV format
    ball_color_lower = (5, 100, 213)  # Lower range of ball color
    ball_color_upper = (30, 171, 255)  # Upper range of ball color

    # Perform blur to reduce noise in the frame
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # Convert to hsv color space
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create a mask to detect the ball within the specified color range
    mask = cv2.inRange(hsv_frame, ball_color_lower, ball_color_upper)

    # Apply morphological operations to remove noise and improve the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours of the ping pong ball
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None

    if len(contours) > 0:
        # Find the largest contourin the mask
        c = max(contours, key=cv2.contourArea)
        # Calculate the minimum enclosing circle and centroid
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Only continue if the radius meets minimum size requirements
        if radius > 5:
            # Draw the circle on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    return center

def main(buffer):
    buffer = int(buffer)
    # List of tracked points
    pts = deque(maxlen=buffer)

    # Initialize video capture object
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the path to a video file
    
    while True:
        # Read from capture
        ret, frame = cap.read()
        
        # Scale frame down to increase FPS
        frame = imutils.resize(frame, width=600)

        # Track ball and return center location
        center = ball_tracker(frame)

        if buffer > 0:
            calculate_trail(pts, buffer, center, frame)

        # Calculate FPS
        calculate_fps(frame)

        # Display the resulting frame with ball detection
        cv2.imshow("Ping Pong Ball Tracker", frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    from argparse import RawTextHelpFormatter
    
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=RawTextHelpFormatter)
    
    parser.add_argument(
        "--buffer",
        help="Size of the buffer that tracks points",
        required=False,
        default="0"
    )

    args = parser.parse_args()

    main(buffer=args.buffer)