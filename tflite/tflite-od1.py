import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def prep_image_tflite(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (320, 320))
    img_resized = img_resized.astype(np.uint8)
    return img_resized

def resize_bbox_to_original(image_path, bbox_resized, target_size=(320, 320)):
    # Read the original image
    original_image = cv2.imread(image_path)
    original_height, original_width = original_image.shape[:2]

    # Calculate the scale factors
    height_scale = original_height / target_size[1]
    width_scale = original_width / target_size[0]

    # Unpack the resized bbox coordinates
    x1, y1, x2, y2 = bbox_resized

    # Scale the bbox back to original image size and convert to (x, y, w, h)
    original_bbox = (
        int(x1 * width_scale),               # Scaled x1
        int(y1 * height_scale),              # Scaled y1
        int((x2 - x1) * width_scale),        # Scaled width
        int((y2 - y1) * height_scale)        # Scaled height
    )

    return original_image, original_bbox

def detect_objects(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], [image])
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

    return boxes, classes, scores, num_detections

def draw_detections(image, boxes, classes, scores, num_detections):
    selected_classes = [33, 32, 29]  # Modify as needed
    best_scores = {cls: 0 for cls in selected_classes}
    best_boxes = {cls: None for cls in selected_classes}

    # Find the highest confidence detection for each class
    for i in range(num_detections):
        cls = int(classes[i])
        if cls in selected_classes and scores[i] > best_scores[cls]:
            best_scores[cls] = scores[i]
            ymin, xmin, ymax, xmax = boxes[i]
            best_boxes[cls] = (int(xmin * 320), int(ymin * 320), int(xmax * 320), int(ymax * 320))

    # Draw the best detections
    for cls in selected_classes:
        if best_scores[cls] > 0:
            startX, startY, endX, endY = best_boxes[cls]
            cx, cy = (startX + endX) // 2, (startY + endY) // 2
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 255), 2)  # Yellow box
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)  # Red centroid
            label = f'Class: {cls}, Confidence: {best_scores[cls]:.2f}, ({cx}, {cy})'
            #label = f'box: {best_boxes[cls]}'
            cv2.putText(image, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return image, [best_boxes[cls] for cls in selected_classes if best_scores[cls] > 0]

# Crop image to bbox
def crop_ball(image, bbox, target_size=(320, 320)):
    x1, y1, x2, y2 = bbox
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img

def black_outside_bbox(image, bbox):
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Ensure bbox coordinates are within the image dimensions
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)

    # Create a mask with the same dimensions as the image
    mask = np.zeros_like(image)

    # Set the region inside the bbox to white (or 1) in the mask
    mask[y1:y2, x1:x2] = 255

    # Apply the mask to black out regions outside the bbox
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

# Feature matching algo
def match_featuresOLD(ref_image, target_image):
    # Resize images to the same dimensions for consistency
    target_image = cv2.resize(target_image, (ref_image.shape[1], ref_image.shape[0]))

    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the images
    ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    target_gray = cv2.GaussianBlur(target_gray, (5, 5), 0)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)

    # FLANN based Matcher
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw matches
    result = cv2.drawMatches(ref_gray, kp1, target_gray, kp2, good_matches, None)
    return result

def match_features(ref_image, target_image):

    # Resize images to the same dimensions for consistency
    target_image = cv2.resize(target_image, (ref_image.shape[1], ref_image.shape[0]))

    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

     # Optionally apply Difference of Gaussian
    #ref_gray = difference_of_gaussian(ref_gray, (1, 1), (87, 87))
    #target_gray = difference_of_gaussian(target_gray, (1, 1), (87, 87))

    # Apply Gaussian blur to smooth the images
    ref_gray = cv2.GaussianBlur(ref_gray, (3, 3), 0)
    target_gray = cv2.GaussianBlur(target_gray, (3, 3), 0)

    # Initiate SIFT detector with modified parameters
    """
    nfeatures: The number of best features to retain. Reducing this number can sometimes help in focusing on the most prominent features.
    nOctaveLayers: The number of layers within each octave in the scale space. Increasing this number can help in detecting more features.
    contrastThreshold: The threshold used to filter out weak features in low-contrast regions of the image. Lowering this threshold may result in detecting more features.
    edgeThreshold: The threshold used to filter out edge-like features. Increase this threshold to retain more edge-like features.
    sigma: The sigma of the Gaussian applied to the input image at the octave #0. If your images are noisy, increasing the sigma might help.
    """
    sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.04, edgeThreshold=10) #500, .04, 10

    # Find keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)

    # FLANN based Matcher
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good_matches.append(m)

    # Perform cross-checking (symmetry test)
    #good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance and n.distance < 0.75 * m.distance]

    # Print out the results
    print(f"Total matches: {len(matches)}")
    print(f"Good matches: {len(good_matches)}")
    for match in good_matches:
        print(f"Distance: {match.distance}")

    # Draw matches
    result = cv2.drawMatches(ref_gray, kp1, target_gray, kp2, good_matches, None)
    return result

# Difference of Gaussian function
def difference_of_gaussian(image, ksize1, ksize2):
    # Apply two Gaussian blurs with different kernel sizes
    blur1 = cv2.GaussianBlur(image, ksize1, 0, borderType=cv2.BORDER_DEFAULT) #CONSTANT, REPLICATE, REFLECT, WRAP, DEFAULT
    blur2 = cv2.GaussianBlur(image, ksize2, 0, borderType=cv2.BORDER_DEFAULT)

    # Subtract the blurs
    dog = cv2.subtract(blur1, blur2)
    return dog

# Function to perform template matching
def template_matching(ref_image, target_image, method=cv2.TM_CCOEFF_NORMED):
    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # Apply template matching
    res = cv2.matchTemplate(target_gray, ref_gray, method)

    # Get the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Determine the top left corner of the matching area
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    # Determine the bottom right corner of the matching area
    w, h = ref_gray.shape[::-1]
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw a rectangle around the matched region
    cv2.rectangle(target_image, top_left, bottom_right, (0, 255, 0), 2)

    return target_image, top_left, bottom_right

# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path='efficientdet.tflite')
interpreter.allocate_tensors()

# Process the first image using object detection to find the ref_image
image = prep_image_tflite('impact1.jpg')
boxes, classes, scores, num_detections = detect_objects(interpreter, image)
output_image, bbox = draw_detections(image.copy(), boxes, classes, scores, num_detections)
cv2.imshow('Detection Output', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


print(f"bbox: {bbox[0]}")
ref_image = crop_ball(image, bbox[0])


# map bbox from 320x320 image to image of original dimensions so feature and template detection works better
orig_image, orig_bbox = resize_bbox_to_original('impact1.jpg', bbox[0])
h_img, w_img, _ = orig_image.shape
x, y, w, h = orig_bbox
x, y, w, h = max(0, x), max(0, y), min(w, w_img - x), min(h, h_img - y)    
# Crop the image to the bounding box
cropped_image = orig_image[y:y+h, x:x+w]


# Process each image using template then feature detection
for image_path in ['impact2.jpg']:
    target_image = cv2.imread(image_path)
    
    # Template detection
    result_image, top_left, bottom_right = template_matching(cropped_image, target_image)

    # Feature detection
    target_cropped = crop_ball(target_image, (top_left[0], top_left[1], bottom_right[0],bottom_right[1]))
    #matched_result = match_features(cropped_image, target_cropped)
    matched_result = match_features(image, target_image)
    cv2.imshow("Feature Matching", matched_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


