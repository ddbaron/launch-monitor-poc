import cv2
import time
import numpy as np
from detect_ball_coco import detect_ball_coco
from picamera2 import Picamera2
from libcamera import controls

def find_still_golf_ball_coco():
    try:
        with Picamera2() as picam2:
            picam2 = configure_vid(picam2)
            picam2.start()

            # Initialize variables
            ball_found = False
            ball_still_start_time = None
            ball_centroid = None
            result = None

            while True:
                frame = picam2.capture_array()
                lux = picam2.capture_metadata()['Lux']
                print(f"Lux is: {lux}.")
                """
                getting these lux values with cam optimized for fast exposure
                9am, indoors
                133 - no LED help
                139 - low LED help
                167 - high LED help
                """

                result = detect_ball_coco(frame)
                        
                cv2.imshow('Find still ball', result['frame'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                #if result is not None and result != {} and ball_found is False:
                if result['object_name'] != 'Not found' and ball_found is False:
                    ball_found = True
                    ball_still_start_time = time.time()
                    
                if ball_found is True:
                    ball_centroid = result['centroid']
                    confidence_percentage = result['confidence'] * 100  # Convert to percentage
                    formatted_percentage = '{:.2f}%'.format(confidence_percentage)
                    print(f"Detected Object: {result['object_name']}, Confidence: {formatted_percentage}, Centroid: {result['centroid']}")
                else:
                    print("Ball not visible right now.")
                
                # Check if the ball has been still for more than x seconds
                if ball_found and time.time() - ball_still_start_time > 5:
                    print("Golf ball found and still!")
                    break
                
            return ball_centroid 

    except Exception as e:
        print(f"Find still ball Error: {e}")
        if str(e) == "Camera __init__ sequence did not complete.":
            picam2.close()
            return None
        return None

    finally:
        cv2.destroyAllWindows()


def configure_vid(picam2):
    try:
        picam2.video_configuration.raw = None
        picam2.video_configuration.size = (1152, 192)
        picam2.video_configuration.controls.FrameRate = 304
        picam2.video_configuration.controls.ExposureTime = 1000
        # picam2.video_configuration.controls.AnalogueGain = 16.0  # 1.0 to 16.0
        # picam2.video_configuration.controls.Brightness = (
        #     0.25  # Floating point number from -1.0 to 1.0. 0.0 is normal
        # )
        # picam2.video_configuration.controls.Contrast = 1.0  # 0.0 to 32.0
        # picam2.video_configuration.controls.Saturation = (
        #     1.0  # 0.0 to 32.0 (0.0 greyscale, 1.0 is normal)
        # )
        # picam2.video_configuration.controls.Sharpness = (
        #     1.0  # Floating point number from 0.0 to 16.0; 1.0 is normal
        # )
        picam2.video_configuration.controls.AeEnable = True
        picam2.video_configuration.controls.AeExposureMode = controls.AeExposureModeEnum.Short # Short / Normal / Long
        picam2.video_configuration.controls.AwbEnable = True
        picam2.video_configuration.controls.AwbMode = (
            controls.AwbModeEnum.Indoor
        )  # Auto, Indoor, Daylight, Cloudy, Fluorescent
        picam2.video_configuration.controls.NoiseReductionMode = (
            controls.draft.NoiseReductionModeEnum.Off
        )
        picam2.configure("video")
        print(f"Video config: {picam2.video_configuration}.")
        return picam2
    except Exception as e:
        print(f"An error occurred configuring Picamera2: {e}")

if __name__ == "__main__":
    
    centroid = find_still_golf_ball_coco()
    if centroid is not None:
        print("picam2 Centroid:", centroid)
    else:
        print("picam2 Unable to find golf ball.")
        

