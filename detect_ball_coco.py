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

def detect_ball_coco(img):
    try:
        # Debugging information

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

        classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

        result = {}

        if classIndex.size > 0 and confidence.size > 0:
            highest_confidence_index = np.argmax(confidence)
            classInd = int(classIndex[highest_confidence_index]-1)
            conf = confidence[highest_confidence_index]
            box = bbox[highest_confidence_index]

            x, y, w, h = box
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            
            # Display the result with a red dot at the centroid
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(img, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Red dot at the centroid
            cv2.putText(img, f"{classLabels[classInd-1]}: ({centroid_x}, {centroid_y})", (x + 10, y + 40),
                cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0), thickness=2)


            result['object_name'] = classLabels[classInd-1]
            result['confidence'] = conf
            result['centroid'] = (centroid_x, centroid_y)
            result['frame'] = img
            
            # Debugging information
            # print(f"Detected Object Index: {classInd}")
            # print(f"Detected Object: {result['object_name']}, Confidence: {result['confidence']}, Centroid: {result['centroid']}")

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
