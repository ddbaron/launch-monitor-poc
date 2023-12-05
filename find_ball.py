import cv2
import time
import numpy as np
from detect_ball_coco import detect_ball_coco
from picamera2 import Picamera2
from libcamera import controls

def find_still_golf_ball_cv2(media_device):
    try:
        cap = cv2.VideoCapture(media_device)
        
        if not cap.isOpened():
            print(f"Error: Unable to open video capture device: {media_device}")
            return None

        # Initialize variables
        ball_found = False
        ball_still_start_time = None
        ball_centroid = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read frame from the video device.")
                if cv2.VideoCapture(media_device_path).isOpened():
                    print("The video device is open but still unable to read frames.")
                else:
                    print("The video device is not open.")
                break

            # Convert the frame to grayscale for better ball detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Use HoughCircles to detect circles (balls) in the frame
            circles = cv2.HoughCircles(
                gray_frame,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=5,
                maxRadius=30
            )

            if circles is not None:
                ball_found = True
                ball_still_start_time = time.time()

                # Draw circles on the frame
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    ball_centroid = (i[0], i[1])

            # Check if the ball has been still for more than 3 seconds
            if ball_found and time.time() - ball_still_start_time > 3:
                print("Golf ball found and still!")
                break

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

       

        return ball_centroid

    except Exception as e:
        print(f"Error: {e}")
        return None
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

def find_still_golf_ball_picam2():
    try:
        picam2 = configure_vid()
        
        picam2.start()

        # Initialize variables
        ball_found = False
        ball_still_start_time = None
        ball_centroid = None

        while True:
            frame = picam2.capture_array()

            # Convert the frame to grayscale for better ball detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            # (5, 5): The size of the kernel or filter. In this case, it's a 5x5 Gaussian kernel. The larger the kernel, the more smoothing is applied to the image.
            #0: The standard deviation (sigma) along the x and y directions of the Gaussian kernel. If set to 0, it is calculated based on the kernel size. A higher value of sigma results in more smoothing.
            blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

            # Apply thresholding to emphasize edges
            _, thresholded_frame = cv2.threshold(
                blurred_frame, 
                40, # threshold value: below will be set to 0 (black), equal to or above set to 255 (white).
                255, # intensity value assigned to pixels that surpass the threshold.
                cv2.THRESH_BINARY
            )
                


            # Use HoughCircles to detect circles (balls) in the frame
            circles = cv2.HoughCircles(
                blurred_frame,
                cv2.HOUGH_GRADIENT, # Detection method (gradient-based
                dp=1,               # [1,2] Inverse ratio of the accumulator resolution to the image resolution
                minDist=300,        # Dist. between centers of circles
                param1=150,         # [50,150] higher threshold passed to Canny edge detector
                param2=30,          # [10,30] Circles with accumulator values above this threshold are returned
                minRadius=35,       # [10,50] min radius of circles to be detected
                maxRadius=80        # [30,100] max radius of circles to be detected
            )

            

            if circles is not None:
                ball_found = True
                ball_still_start_time = time.time()

                # Draw circles on the frame
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    ball_centroid = (i[0], i[1])
                    cv2.circle(frame, ball_centroid, 5, (0, 0, 255), -1)
                    # Display XYZ coordinates on the screen
                    coordinates_text = "XYZ: ({:.2f}, {:.2f})".format(ball_centroid[0], ball_centroid[1])
                    cv2.putText(frame, coordinates_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Check if the ball has been still for more than x seconds
            if ball_found and time.time() - ball_still_start_time > 0.5:
                print("Golf ball found and still!")
                cv2.imwrite('found_ball.jpg', frame);
                break

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

       

        return ball_centroid

    except Exception as e:
        print(f"Error: {e}")
        return None
        
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

def find_still_golf_ball_picam3():
    
    try:
        picam2 = configure_vid()
        
        picam2.start()

        # Initialize variables
        ball_found = False
        ball_still_start_time = None
        ball_centroid = None

        while True:
            resized_frame = picam2.capture_array()

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

            # Improved object detection with circle and dot drawing
            for contour in contours:
                # Filter contours based on area (may need to adjust the threshold)
                if cv2.contourArea(contour) > 100:
                    
                    ball_found = True
                    ball_still_start_time = time.time()
                
                    # Fit a circle around the contour
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    ball_centroid = center = (int(x), int(y))

                    # Draw the circle and dot
                    cv2.circle(resized_frame, center, int(radius), (0, 255, 0), 2)
                    cv2.circle(resized_frame, center, 5, (0, 0, 255), -1)

                    # Display XYZ coordinates on the screen
                    coordinates_text = "XYZ: ({:.2f}, {:.2f})".format(center[0], center[1])
                    cv2.putText(resized_frame, coordinates_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            

            # Check if the ball has been still for more than 3 seconds
            if ball_found and time.time() - ball_still_start_time > 10:
                print("Golf ball found and still for 3 seconds!")
                break

            cv2.imshow('Frame', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

       

        return ball_centroid

    except Exception as e:
        print(f"Error: {e}")
        return None
        
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

def find_still_golf_ball_coco():
    try:
        picam2 = configure_vid()
        picam2.start()

        # Initialize variables
        ball_found = False
        ball_still_start_time = None
        ball_centroid = None
        result = None

        while True:
            frame = picam2.capture_array()

            result = detect_ball_coco(frame)
            
            if result is not None and result != {} and ball_found is False:
                ball_found = True
                ball_centroid = result['centroid']
                ball_still_start_time = time.time()
                print(f"ball still started at: {ball_still_start_time}")

            # Check if the ball has been still for more than x seconds
            print(f"time().time is: {time.time()}")
            if ball_found and time.time() - ball_still_start_time > 5:
                print("Golf ball found and still!")
                cv2.imwrite('found_ball.jpg', frame);
                break

            if result['frame'] is not None:
                frame = result['frame']
                
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            if result is not None and result != {}:
                confidence_percentage = result['confidence'] * 100  # Convert to percentage
                formatted_percentage = '{:.2f}%'.format(confidence_percentage)
                print(f"Detected Object: {result['object_name']}, Confidence: {formatted_percentage}, Centroid: {result['centroid']}")

            else:
                print("Ball not found by coco.")
            
        return ball_centroid 

    except Exception as e:
        print(f"Error: {e}")
        return None

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

def configure_vid():
    picam2 = Picamera2()

    picam2.video_configuration.raw = None
    picam2.video_configuration.size = (1152, 192)
    picam2.video_configuration.controls.FrameRate = 304
    picam2.video_configuration.controls.ExposureTime = 1000
    # picam2.video_configuration.controls.AnalogueGain = 16.0  # 1.0 to 16.0
    # picam2.video_configuration.controls.Brightness = (
        # 0.25  # Floating point number from -1.0 to 1.0. 0.0 is normal
    # )
    # picam2.video_configuration.controls.Contrast = 1.0  # 0.0 to 32.0
    # picam2.video_configuration.controls.Saturation = (
        # 1.0  # 0.0 to 32.0 (0.0 greyscale, 1.0 is normal)
    # )
    # picam2.video_configuration.controls.Sharpness = (
        # 1.0  # Floating point number from 0.0 to 16.0; 1.0 is normal
    # )
    picam2.video_configuration.controls.AeEnable = True
    picam2.video_configuration.controls.AeExposureMode = controls.AeExposureModeEnum.Short # Short / Normal / Long
    picam2.video_configuration.controls.AwbEnable = False
    picam2.video_configuration.controls.AwbMode = (
        controls.AwbModeEnum.Indoor
    )  # Auto, Indoor, Daylight, Cloudy, Fluorescent
    picam2.video_configuration.controls.NoiseReductionMode = (
        controls.draft.NoiseReductionModeEnum.Off
    )
    picam2.configure("video")
    print(f"Video config: {picam2.video_configuration}.")
    return picam2

if __name__ == "__main__":
    
    #print(f"CV2 version: {cv2.__version__}")
    media_device_path = '/dev/video0'  # Replace with the actual path to your video device
    
    #centroid = find_still_golf_ball_cv2(media_device_path)
    # if centroid is not None:
        # print("CV2 Centroid:", centroid)
    # else:
        # print("CV2 Unable to find golf ball.")
        
    #centroid = find_still_golf_ball_picam2()
    centroid = find_still_golf_ball_coco()
    if centroid is not None:
        print("picam2 Centroid:", centroid)
    else:
        print("picam2 Unable to find golf ball.")
        

