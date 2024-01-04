from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput
from libcamera import controls
import time
import cv2
import numpy as np
import sys
import os


width = 1152
height = 192
fps = 304
shutter = 10000

picam2 = Picamera2()

picam2.video_configuration.raw = None
picam2.video_configuration.size = (width, height)
#picam2.video_configuration.format = "RGB888"
picam2.video_configuration.controls.FrameRate = fps
picam2.video_configuration.controls.ExposureTime = shutter
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
    
frame_buffer = []
frame_counter = 0
    
picam2.start()

while True:
    frame = picam2.capture_array() #format is XBGR8888
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (width, height))
    if frame_counter == 0:
        print(f"Frame Array Shape: {resized_frame.shape}")
        print(f"Frame Dimensions: {resized_frame.ndim}")
        print(f"Frame datatype: {resized_frame.dtype}")
    
    frame_buffer.append(resized_frame.copy())
    
    frame_counter += 1
    if frame_counter > 1000:
        break
    
picam2.stop()

output_filename = 'vid.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
hheight, wwidth = frame_buffer[0].shape[:2]

# Create a VideoWriter object
try:
    out = cv2.VideoWriter(output_filename, fourcc, fps, (wwidth, hheight))
except Exception as e:
    print("Exception:", e)
if not out.isOpened():
    print("Error: VideoWriter not opened successfully.")

print("Output Filename:", output_filename)
print("FourCC:", fourcc)
print("FPS:", fps)
print("Width:", wwidth)
print("Height:", hheight)

print("Number of Frames in Buffer:", len(frame_buffer))
print("VideoWriter Configuration:", out.getBackendName())

# Write each processed frame to the video file
for idx, frame in enumerate(frame_buffer):
    bgr_frame = frame[:, :, [2, 1, 0, 3]] # assuming XBGR8888
    #bgr_frame = frame[:,:,0:3]
    print("Frame Shape:", bgr_frame.shape)
    print(f"Expected width: {wwidth}, heigh: {hheight}")
    print("Data Type:", bgr_frame.dtype)

    # Display the frame
    cv2.imshow('Frame', bgr_frame)

    # Write the frame to the video
    # try:
    #     if out.write(bgr_frame):
    #         continue
    #     else:
    #         print(f"Failed to write frame {idx}")
    #         #break
    # except Exception as e:
    #     print(f"Exception at frame {idx}:", e)


if out.isOpened():
    print("Video writing completed successfully.")
else:
    print("Error: Video writing failed.")

# Release the VideoWriter object
out.release()
# Close the OpenCV window
cv2.destroyAllWindows()

