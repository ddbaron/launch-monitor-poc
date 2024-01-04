import cv2
import os
import time
import logging

# Configuration
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
VIDEO_PATH = "/dev/shm/output.h264"

# Setup logging
logging.basicConfig(level=logging.INFO)  # Can change to DEBUG, WARNING, etc.

# Load class names
class_names = []
try:
    with open("yolo/coco-classes.txt", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
        #print(f"Class names: {class_names}")
except FileNotFoundError:
    logging.error("Class names file not found")
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
    net = cv2.dnn.readNet("yolo/yolov4-tiny.weights", "yolo/yolov4-tiny.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # Uncomment below for CUDA backend if available
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
except cv2.error as e:
    logging.error(f"Error loading model: {e}")
    exit()

# Processing frames
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        logging.info("No more frames to read")
        break

    try:
        start = time.time()
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()

        # Check if detections are made
        if classes is not None and len(classes) > 0:
            start_drawing = time.time()

            for (classid, score, box) in zip(classes, scores, boxes):
                color = COLORS[int(classid) % len(COLORS)]
                label = f"{class_names[int(classid)]}: {score:.2f}"
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            end_drawing = time.time()

            fps_label = f"FPS: {1 / (end - start):.2f} (excluding drawing time of {(end_drawing - start_drawing) * 1000:.2f}ms)"
            logging.info(f"fps_label: {fps_label}")
            
            cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Display the frame even if no detections are made
        cv2.imshow("detections", frame)
    except cv2.error as e:
        logging.error(f"Error processing frame: {e}")
