"""
LM3
    requires: media_ctl.py, find_ball.py->detect_ball_coco.py

**prework
	call media-ctl to configure camera for high fps mode
**get ready
	use cv2 to detect presence of a ball sitting relatively still. identify the x.y for crop later
	stop cv2
**ready
	call media-ctl, crop for the x.y location
    call libcamera-vid record video with --circular buffer
		libcamera-vid --circular 1 --inline ... producing an .h264 file
	wait for the click sound of a hit, signal licamera-vid to exit, remember the sound_ts
	crop the .h264 file using the sound_ts:
		ffmpeg -ss <start_timestamp> -i input.h264 -t 2 -c copy output.h264
	ffmpeg the output.h264 file to output.mp4
**postwork
	use cv2 to analyze shot, and find ball xyz
	produce mp4 video from .h264 using ffmpeg, showing impact with centroid and xyz data overlay
	scp the vid to macbook for review
	

"""
import cv2
import time
import argparse
import numpy as np
import sys
from media_ctl import set_v4l2_format, list_cameras, calculate_offset
from find_ball import find_still_golf_ball_coco, configure_vid
from picamera2 import Picamera2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Monitor 3")
    parser.add_argument("--width", type=int, default=1152, help="Desired width")
    parser.add_argument("--height", type=int, default=192, help="Desired height")

    args = parser.parse_args()

    # PREWORK

    # Call media-ctl to configure camera for high fps mode
    successful_device = set_v4l2_format(args.width, args.height)
    if successful_device:
        print(
            f"Successful configuration on device: {successful_device}, listing cameras:"
        )
        list_cameras()
    else:
        print("Camera configuration unsuccessful")
        sys.exit()

    # Use cv2 to detect the presence of a ball sitting relatively still. Identify the x.y for crop later
    centroid = find_still_golf_ball_coco()
    if centroid is not None:
        print("picam2 Centroid:", centroid)
        offset_x, offset_y, centroid_x, centroid_y = calculate_offset(
            crop_w=args.width,
            crop_h=args.height,
            centroid_x=centroid[0],
            centroid_y=centroid[1],
        )
    else:
        print("picam2 Unable to find golf ball.")
        sys.exit()

    # READY -

    # Call media-ctl to configure camera for high fps mode
    print(f"Setting centroid x:{centroid_x} and y: {centroid_y} ")
    successful_device = set_v4l2_format(args.width, args.height, centroid_x, centroid_y)
    if successful_device:
        print(
            f"Successful configuration on device: {successful_device}, listing cameras:"
        )
        list_cameras()
        with Picamera2() as picam2:
            picam2 = configure_vid(picam2)
            picam2.start()
            while True:
                frame = picam2.capture_array()
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            picam2.stop()
            cv2.destroyAllWindows()
    else:
        print("Camera configuration unsuccessful")
        sys.exit()
