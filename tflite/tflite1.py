"""
abandonded as this was for image classification not object detection
"""

import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

def load_model(model_path):
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_image(image_path, input_shape):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found at path: {image_path}")
        img_resized = cv2.resize(img, input_shape)
        img_display = np.squeeze(img_resized).astype(np.uint8)  # For display purposes
        cv2.imshow('Preprocessed image: ', img_display)
        cv2.waitKey(0)  # Waits for a key press to close the displayed image
        img = np.expand_dims(img_resized, axis=0).astype(np.float32)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def detect_objects(interpreter, image):
    try:
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], image)
        print(f"input_details[0]['index']: {input_details[0]['index']}")
        interpreter.invoke()
        output_details = interpreter.get_output_details()
        outdetail = interpreter.get_tensor(output_details[0]['index'])
        print(f"outdetail: {outdetail}")
        boxes = outdetail[0]  # Bounding box coordinates
        return boxes
    except Exception as e:
        print(f"Error during object detection: {e}")
        return None

def draw_boxes(image_path, boxes):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found at path: {image_path}")
        for box in boxes:
            ymin, xmin, ymax, xmax = box
            (startX, startY, endX, endY) = (xmin * img.shape[1], ymin * img.shape[0], xmax * img.shape[1], ymax * img.shape[0])
            cv2.rectangle(img, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)
            cx = int((startX + endX) / 2)
            cy = int((startY + endY) / 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        return img
    except Exception as e:
        print(f"Error drawing boxes: {e}")
        return None

# Main execution
model_path = 'EfficientNetB3Balls93.86.tflite'
image_path = 'impact1.jpg'
interpreter = load_model(model_path)

if interpreter is not None:
    input_details = interpreter.get_input_details()
    input_shape = (input_details[0]['shape'][2], input_details[0]['shape'][1])
    image = load_image(image_path, input_shape)
    if image is not None:
        boxes = detect_objects(interpreter, image)
        # if boxes is not None:
        #     output_img = draw_boxes(image_path, boxes)
        #     if output_img is not None:
        #         cv2.imwrite('output.jpg', output_img)
