# library for using libcamera-vid via picamera2
#
# video_config = picam2.create_video_configuration(main={"size": (1920, 1080), 'format': 'YUV420'},
#                                                 controls={'FrameRate': 50, 'NoiseReductionMode': 1})
# https://github.com/raspberrypi/picamera2/issues/737#issuecomment-1630382707


from picamera2 import Picamera2, Preview
import time
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
picam2.start_preview(Preview.QTGL)
picam2.start()
time.sleep(2)
picam2.capture_file("test.jpg")
