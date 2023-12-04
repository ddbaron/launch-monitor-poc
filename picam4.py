from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput
from libcamera import controls
import time
import cv2
import numpy as np

picam2 = Picamera2()

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

# Capture frames during recording for 'duration' seconds
start_time = time.time()
duration = 3  # seconds

frame_buffer = []
cnt = 0

encoder = H264Encoder(repeat=True, iperiod=15)
output = FileOutput('foo.h264')
picam2.start()

while time.time() - start_time < duration:
    # Capture a frame during recording
    frame = picam2.capture_array()
    if cnt == 0:
        picam2frame = frame.copy()
        cnt += 1

    frame_buffer.append(frame.copy())

    # For example, you can use cv2.imshow to display the frame
    #cv2.imshow("Recording Frame", frame)
    #cv2.waitKey(1)  # Update display

print(f"we have {len(frame_buffer)} frames.")

# Create a VideoWriter object
height, width, _ = frame_buffer[0].shape
out = cv2.VideoWriter(
    "vid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height), True
)

for idx, frame in enumerate(frame_buffer):
    success = out.write(frame)
    if not success:
        print(f"Error writing frame {idx}, aborting.")
        if np.array_equal(frame, picam2frame):
            print("First frames are the same")
        else:
            print("First frames are NOT the same")
        break

out.release()
picam2.stop()
cv2.destroyAllWindows()  # Close the display window
