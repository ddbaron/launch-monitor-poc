import cv2
import os
import time
import logging
import numpy as np

# Configuration
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
VIDEO_PATH = "/dev/shm/output.h264"
CLASS_NAMES_PATH = "yolo/coco-classes.txt"
MODEL_WEIGHTS_PATH = "yolo/yolov4-tiny.weights"
MODEL_CONFIG_PATH = "yolo/yolov4-tiny.cfg"

# Setup logging
logging.basicConfig(level=logging.DEBUG)  # Can change to DEBUG, WARNING, etc.

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
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
except cv2.error as e:
    logging.error(f"Error loading model: {e}")
    exit()


def pad_to_square(frame):
    """
    Pads the input frame to make it square.
    """
    height, width = frame.shape[:2]
    # Determine the difference between the longer and shorter side
    delta = abs(height - width)
    top, bottom, left, right = 0, 0, 0, 0

    if height > width:
        # If height is greater, pad left and right
        left = delta // 2
        right = delta - left
    else:
        # If width is greater, pad top and bottom
        top = delta // 2
        bottom = delta - top

    # Pad with black color (0, 0, 0)
    padded_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_frame

# Processing frames
while cv2.waitKey(1) < 1:
    (grabbed, resized_frame) = vc.read()
    if not grabbed:
        logging.info("No more frames to read")
        break

    # Pad the frame to square
    #square_frame = pad_to_square(resized_frame)
    # Resize to 416x416 for YOLO input
    #resized_frame = cv2.resize(square_frame, (416, 416))

    try:
        start = time.time()
        classes, scores, boxes = model.detect(resized_frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()

        if classes is not None and len(classes) > 0:
            start_drawing = time.time()

            # Find the detection with the highest score
            max_score_index = np.argmax(scores)
            classid = classes[max_score_index]
            score = scores[max_score_index]
            box = boxes[max_score_index]

            logging.debug(f"Max confidence classid: {classid}")
            logging.debug(f"Max confidence score: {score}")
            logging.debug(f"Max confidence box: {box}")

            color = COLORS[int(classid) % len(COLORS)]
            label = f"{class_names[int(classid)]}: {score:.2f}"

            # Green circle around ball with red dot in center
            x, y, w, h = box
            center_x = x + w // 2
            center_y = y + h // 2
            radius = min(w, h) // 2
            cv2.circle(resized_frame, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.circle(resized_frame, (center_x, center_y), 2, (0, 0, 255), -1)
            cv2.putText(resized_frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            end_drawing = time.time()


            fps_label = f"FPS: {1 / (end - start):.2f} (excluding drawing time of {(end_drawing - start_drawing) * 1000:.2f}ms)"
            logging.debug(f"fps_label: {fps_label}")  # Changed to debug level
            cv2.putText(resized_frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("detections", resized_frame)
    except cv2.error as e:
        logging.error(f"Error processing frame: {e}")

# Release resources
vc.release()
cv2.destroyAllWindows()
