import cv2
import numpy as np
import sys
import os

# Check if a video path argument is provided
if len(sys.argv) < 2:
    print("No video path provided. Using default path '/dev/shm/tst.h264'")
    video_path = '/dev/shm/tst.h264'
else:
    # Get the video path from the command-line argument
    video_path = sys.argv[1]

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the width and height of the video frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Extract filename and extension
filename, extension = os.path.splitext(os.path.basename(video_path))

# Create a VideoWriter object with the same filename but append '_CV'
output_filename = f'{filename}_CV_{width}x{height}.mp4'
out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

# Known real-world diameter of the golf ball in millimeters
real_world_diameter_mm = 42.67

# Estimated focal length in pixels (this value needs to be obtained through camera calibration)
focal_length = 1000  # Example value, replace with the actual focal length

while True:
    # Read the next frame
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (width, height))

    # Color-based thresholding for the golf ball (adjust the range based on the golf ball color)
    lower_color = np.array([20, 100, 100])
    upper_color = np.array([40, 255, 255])
    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    # Apply morphological operations to clean up the binary image
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Example: Improved object detection with circle and dot drawing
    for contour in contours:
        # Filter contours based on area (you may need to adjust the threshold)
        if cv2.contourArea(contour) > 100:
            # Fit a circle around the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))

            # Draw the circle and dot
            cv2.circle(resized_frame, center, int(radius), (0, 255, 0), 2)
            cv2.circle(resized_frame, center, 5, (0, 0, 255), -1)

            # Calculate the depth (z-coordinate) based on the real-world diameter
            depth = (focal_length * real_world_diameter_mm) / (2 * radius)

            # Display XYZ coordinates on the screen
            coordinates_text = "XYZ: ({:.2f}, {:.2f}, {:.2f})".format(center[0], center[1], depth)
            cv2.putText(resized_frame, coordinates_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Check if part of the object is off-screen (adjust the margin as needed)
            margin = 20
            if center[0] - radius < margin or center[0] + radius > width - margin or \
               center[1] - radius < margin or center[1] + radius > height - margin:
                print("Object partially off-screen. Stopping detection.")
                continue

            # Output a message with real-world coordinates
            print("Golf ball detected at real-world coordinates (x, y, z): ({:.2f}, {:.2f}, {:.2f})".format(center[0], center[1], depth))

    # Write the processed frame to the output video
    out.write(resized_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

