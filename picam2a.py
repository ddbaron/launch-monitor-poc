from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput
from libcamera import controls
import time
import cv2
import numpy as np
import sys
import os

def configure_camera():
    """
    Configure Picamera2 with specific video settings.
    """
    picam2 = Picamera2()
    picam2.video_configuration.raw = None
    picam2.video_configuration.size = (1152, 192)
    #picam2.video_configuration.format = "RGB888"
    picam2.video_configuration.controls.FrameRate = 304
    picam2.video_configuration.controls.ExposureTime = 1000
    picam2.video_configuration.controls.AnalogueGain = 16.0  # 1.0 to 16.0
    picam2.video_configuration.controls.Brightness = 0.25  # -1.0 to 1.0; 0.0 is normal
    picam2.video_configuration.controls.Contrast = 1.0  # 0.0 to 32.0
    picam2.video_configuration.controls.Saturation = 1.0  # 0.0 to 32.0 (0.0 greyscale, 1.0 is normal)
    picam2.video_configuration.controls.Sharpness = 1.0  # 0.0 to 16.0; 1.0 is normal
    picam2.video_configuration.controls.AeEnable = False
    picam2.video_configuration.controls.AeExposureMode = controls.AeExposureModeEnum.Short
    picam2.video_configuration.controls.AwbEnable = False
    picam2.video_configuration.controls.AwbMode = controls.AwbModeEnum.Indoor  # Auto, Indoor, Daylight, Cloudy, Fluorescent
    picam2.video_configuration.controls.NoiseReductionMode = controls.draft.NoiseReductionModeEnum.Off
    picam2.configure("video")
    print(f"Video configuration: {picam2.video_configuration}")
    return picam2

def record_video(picam2, duration=5):
    """
    Record video using Picamera2 for a specified duration.
    """
    encoder = H264Encoder()
    output = FfmpegOutput("test.mp4")

    picam2.start_recording(encoder, output, quality=Quality.HIGH)
    time.sleep(duration)
    picam2.stop_recording()

def record_impact(picam2):
    """
    detect the golf ball, find centroid
    monitor the centroid until it leaves the frame
    write the last 50 frames to an mp4 file 
    """
    # Variables for capturing frames until the ball leaves the frame
    frames_before_exit = -500

    # Lists to store frames and timestamps
    frame_buffer = []
    frame_counter = 0;
    frame_out_counter = 0;
    
    out = cv2.VideoWriter('ball.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1152, 192))
    
    picam2.start()

    while True:
        frame = picam2.capture_array() #format is XBGR8888
        print(f"Frame Array Shape: {frame.shape}")
        print(f"Frame Dimensions: {frame.ndim}")
        print(f"Frame datatype: {frame.dtype}")

        # Print the elements of the XBGR8888 array
        #for pixel in frame:
            #print(f"Blue: {pixel[0]}, Green: {pixel[1]}, Red: {pixel[2]}, eXtra: {pixel[3]}")

        # Increment the counter
        frame_counter += 1
        
        bgr_image = frame[:,:,0:3] 
        success = out.write(bgr_image)
        if not success:
            print(f"Error writing frame {frame_counter}, aborting.")
            break
        else: 
            frame_out_counter += 1
        
        # stop after x frames
        if frame_counter > 500:
            break
 
    # Release resources
    out.release()
    picam2.stop()
    cv2.destroyAllWindows()
    
    # Print the total number of frames
    total_frames = len(frame_buffer)
    print(f"Total frame_buffer: {total_frames} Total frame_counter: {frame_counter}.") 
    print(f"Actual frames written: {frame_out_counter}.")

if __name__ == "__main__":
    # Configure camera with specific settings
    camera = configure_camera()
    
    record_impact(camera)

    # Record video for 5 seconds
    #record_video(camera, duration=5)
