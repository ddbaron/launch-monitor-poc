import cv2
import os
import time
import logging
import numpy as np

# Configuration
CONFIDENCE_THRESHOLD = 0.1 # Only detections with confidence scores higher than this threshold are returned. A higher value decreases the likelihood of false positives but may miss some true detections.
NMS_THRESHOLD = 0.4 # If the overlap between boxes is higher than this threshold, the box with the lower confidence score is removed.
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
VIDEO_PATH = "/dev/shm/output.h264"
CLASS_NAMES_PATH = "yolo/coco-classes.txt"
MODEL_WEIGHTS_PATH = "yolo/yolov4-tiny.weights"
MODEL_CONFIG_PATH = "yolo/yolov4-tiny.cfg"
LOGGING = logging.DEBUG # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Setup logging
logging.basicConfig(level=LOGGING)  

# Load class names
class_names = []
try:
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
except Exception as e:  # Broad exception handling for any file reading errors
    logging.error(f"Error reading class names file: {e}")
    exit()

# Initialize video capture
if not os.path.exists(VIDEO_PATH):
    logging.error(f"Video file not found: {VIDEO_PATH}")
    exit()
try:
    vc = cv2.VideoCapture(VIDEO_PATH)
    if not vc.isOpened():
        raise IOError("Cannot open video file")
except IOError as e:
    logging.error(f"Error opening video source: {e}")
    exit()

# Load YOLOv4-tiny model
try:
    net = cv2.dnn.readNet(MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH)
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
except cv2.error as e:
    logging.error(f"Error loading model: {e}")
    exit()

def stretch_box(box):
    # Assume box is in the format [x, y, width, height]
    x, y, w, h = box

    # Calculate the increase in size
    increase_w = w * 0.15
    increase_h = h * 0.15

    # Adjust the starting point (x, y) to move it up and to the left
    new_x = x - increase_w / 2
    new_y = y - increase_h / 2

    # Increase the width and height
    new_w = w + increase_w
    new_h = h + increase_h

    # New stretched box
    stretched_box = [int(new_x), int(new_y), int(new_w), int(new_h)]
    return stretched_box

def detect_circle(image, bbox):
    # Extract the region of interest (ROI) using YOLO detection bounding box
    x, y, w, h = bbox
    roi = image[y:y+h,x:x+w]

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Optional: Apply Gaussian blur to reduce noise
    # The values 7x7 indicate the width and height of the kernel used for the blur. A larger kernel size will result in more blurring.
    #gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 3x3, 5x5, 7x7 etc....

    # Apply Hough Circle Transform
    '''
        dp (1): The inverse ratio of the accumulator resolution to the image resolution. Here, it's set to 1, meaning the accumulator has the same resolution as the input image.
        minDist (20): The minimum distance between the centers of detected circles. Here, it's 20 pixels. This helps in avoiding multiple nearby circles being mistakenly detected as one.
        param1 (50): The higher threshold for the internal Canny edge detector. param2 is set at twice this value.
        param2 (30): The accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected.
        minRadius (0) and maxRadius (0): The minimum and maximum radius of the circles to be detected. Here, both are set to 0, indicating that there is no specific limit.
    '''
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, int(w/3),
                               param1=40, param2=20, minRadius=int(h/3), maxRadius=int(w/2))

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Adjust circle coordinates to match original image
            adjusted_circle_center = (i[0] + x, i[1] + y)
            adjusted_circle_radius = i[2]

            # Draw the outer circle and center of the circle on the original image
            cv2.circle(image, adjusted_circle_center, adjusted_circle_radius, (0, 255, 0), 2)
            cv2.circle(image, adjusted_circle_center, 2, (0, 0, 255), 3)

    # Return the image with the detected circle
    return image

# Processing frames
while cv2.waitKey(1) < 1:
    (grabbed, resized_frame) = vc.read()
    if not grabbed:
        logging.info("No more frames to read")
        break
  
    # Configurable list of class IDs to be handled
    allowed_classids = [32, 29]  # Example class IDs for 'sports ball', 'frisbee', etc.

    # preprocess image
    alpha = 2.0 # Contrast control
    beta = 16    # Brightness control
    blurred = cv2.GaussianBlur(resized_frame, (3, 3), 0)
    adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta) 
    resized_frame = adjusted

    try:
        start = time.time()
        classes, scores, boxes = model.detect(resized_frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()

        if classes is not None and len(classes) > 0:
            for classid, score, box in zip(classes, scores, boxes):
                if classid in allowed_classids:
                    # Process only if classid is in allowed_classids
                    color = COLORS[int(classid) % len(COLORS)]
                    label = f"{class_names[int(classid)]}: {score:.2f}"
                    logging.debug(f"label: {label} box: {box}")
                    cv2.rectangle(resized_frame, box, color, 2)
                    # Use Hough Circle Transform to accurately draw a circle around the ball found by yolo
                    resized_frame = detect_circle(resized_frame, stretch_box(box))
                    
                    cv2.putText(resized_frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("detections", resized_frame)
    except cv2.error as e:
        logging.error(f"Error processing frame: {e}")

# Release resources
vc.release()
cv2.destroyAllWindows()
