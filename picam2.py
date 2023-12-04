from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput
from libcamera import controls
import time

picam2 = Picamera2()

picam2.video_configuration.raw = None
picam2.video_configuration.size = (1152, 192)
picam2.video_configuration.controls.FrameRate = 304
picam2.video_configuration.controls.ExposureTime = 1000
picam2.video_configuration.controls.AnalogueGain = 16.0 # 1.0 to 16.0
picam2.video_configuration.controls.Brightness = 0.25 # Floating point number from -1.0 to 1.0. 0.0 is normal
picam2.video_configuration.controls.Contrast = 1.0 # 0.0 to 32.0
picam2.video_configuration.controls.Saturation = 1.0 # 0.0 to 32.0 (0.0 greyscale, 1.0 is normal)
picam2.video_configuration.controls.Sharpness = 1.0 # Floating point number from 0.0 to 16.0; 1.0 is normal
picam2.video_configuration.controls.AeEnable = False
picam2.video_configuration.controls.AeExposureMode = controls.AeExposureModeEnum.Short
picam2.video_configuration.controls.AwbEnable = False
picam2.video_configuration.controls.AwbMode = controls.AwbModeEnum.Indoor #Auto, Indoor, Daylight, Cloudy, Fluorescent
picam2.video_configuration.controls.NoiseReductionMode = controls.draft.NoiseReductionModeEnum.Off
picam2.configure("video")
print(f"Video config: {picam2.video_configuration}.")

encoder = H264Encoder()
output = FfmpegOutput("test.mp4")

picam2.start_recording(encoder, output, quality=Quality.HIGH)
time.sleep(5)

picam2.stop_recording()
