"""
    Used by find_ball.py
    Also works standalone
"""
import cv2
import numpy as np

# Initialize the model outside the function to avoid reloading it every time
config_file = 'coco/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'coco/frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
filename = 'coco/labels.txt'
with open(filename, 'rt') as spt:
    classLabels = spt.read().rstrip('\n').split('\n')
    print("Established labels for object detection")

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

font = cv2.FONT_HERSHEY_PLAIN

def detect_ball_coco(img):
    try:
        result={}
        # Convert to RGB format if needed
        if img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_height, img_width, _ = img.shape
        desired_input_size = 320

        # Adjust input size while maintaining the aspect ratio
        if img_height > img_width:
            model.setInputSize(desired_input_size, int(desired_input_size * img_height / img_width))
        else:
            model.setInputSize(int(desired_input_size * img_width / img_height), desired_input_size)

        classIndex, confidence, bbox = model.detect(img, confThreshold=0.7)

        # # After getting classIndex, confidence, and bbox
        # print(f"Length of classIndex: {len(classIndex)} and classIndex is: {classIndex}")
        # print(f"Length of confidence: {len(confidence)} and confidence is: {confidence} ")
        # print(f"Length of bbox: {len(bbox)} and bbox is: {bbox}")
            
        if classIndex is not None and len(classIndex) > 0:
            classInd = int(classIndex[0])  # the first value is always the highest confidence
            conf = confidence[0]
            box = bbox[0]

            # Display a red dot at the centroid
            centroid_x = box[0] + box[2] // 2
            centroid_y = box[1] + box[3] // 2
            cv2.circle(img, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Red dot at the centroid

            # Surround with a bounding box rectangle
            cv2.rectangle(img, box, (255, 0, 0), 2)

            # Display the object name and confidence
            cv2.putText(img, f"{classLabels[classInd-1]}: {conf:.2f}", (box[0] + 10, box[1] + 40),
                        cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0), thickness=2)

            # Populate the result object
            result['object_name'] = classLabels[classInd-1]
            result['confidence'] = conf
            result['centroid'] = (centroid_x, centroid_y)
            result['frame'] = img  # overlayed now with cv2 data


        else:
            # No object detected
            result['object_name'] = "Not found"
            result['confidence'] = 0
            result['centroid'] = (0, 0)
            result['frame'] = img
        
        # print("Result values:")
        # print(f"Object Name: {result['object_name']}")
        # print(f"Confidence: {result['confidence']}")
        # print(f"Centroid: {result['centroid']}")
        return result
    except Exception as e:
        print(f"Error in detect_ball_coco: {e}")
        return None

if __name__ == "__main__":
    img = cv2.imread('found_ball.jpg')
    result = detect_ball_coco(img)

    if result:
        print(f"Detected Object: {result['object_name']}, Confidence: {result['confidence']}, Centroid: {result['centroid']}")
    else:
        print("No object detected.")
