#!/usr/bin/python3
import time

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import CircularOutput
from libcamera import controls
import time
import cv2
import numpy as np

picam2 = Picamera2()
fps = 304
dur = 5
buffersize=int(fps * (dur + 0.2))

picam2.video_configuration.raw = None
picam2.video_configuration.size = (1152, 192)
picam2.video_configuration.controls.FrameRate = 304
picam2.video_configuration.controls.ExposureTime = 1000
picam2.video_configuration.controls.AnalogueGain = 16.0  # 1.0 to 16.0
picam2.video_configuration.controls.Brightness = (
    0.25  # Floating point number from -1.0 to 1.0. 0.0 is normal
)
picam2.video_configuration.controls.Contrast = 1.0  # 0.0 to 32.0
picam2.video_configuration.controls.Saturation = (
    1.0  # 0.0 to 32.0 (0.0 greyscale, 1.0 is normal)
)
picam2.video_configuration.controls.Sharpness = (
    1.0  # Floating point number from 0.0 to 16.0; 1.0 is normal
)
picam2.video_configuration.controls.AeEnable = False
picam2.video_configuration.controls.AeExposureMode = controls.AeExposureModeEnum.Short
picam2.video_configuration.controls.AwbEnable = False
picam2.video_configuration.controls.AwbMode = (
    controls.AwbModeEnum.Indoor
)  # Auto, Indoor, Daylight, Cloudy, Fluorescent
picam2.video_configuration.controls.NoiseReductionMode = (
    controls.draft.NoiseReductionModeEnum.Off
)
picam2.configure("video")
print(f"Video config: {picam2.video_configuration}.")

buffersize=int(fps * (dur + 0.2))

encoder = H264Encoder()
output = CircularOutput(buffersize=50, file=None, outputtofile=False)
picam2.start_encoder(encoder, output)
print("started encoder, waiting 10 seconds")
time.sleep(10)
output.fileoutput = "file.h264"
picam2.start()
print("started recording?, waiting 2 seconds")
time.sleep(2)
picam2.stop()
picam2.stop_encoder()
cap = cv2.VideoCapture('file.h264')
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
	total_frames = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		total_frames +=1
	print(f"Total frames in the video: {total_frames}")
	cap.release()



