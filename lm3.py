"""
LM3

**prework
	call media-ctl to configure camera for high fps mode
**get ready
	use cv2 to detect presence of a ball sitting relatively still. identify the x.y for crop later
	stop cv2
**ready
	call libcamera-vid, crop for the x.y location, record video with --circular buffer
		libcamera-vid --circular 1 --inline ...
	wait for the click sound of a hit, signal licamera-vid to exit, remember the sound_ts
	crop the file using the sound_ts
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
from media_ctl import set_v4l2_format, list_cameras 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Monitor 3")
    parser.add_argument("--width", type=int, default=1152, help="Desired width")
    parser.add_argument("--height", type=int, default=192, help="Desired height")

    args = parser.parse_args()

    successful_device = set_v4l2_format(args.width, args.height)
    if successful_device:
        print(f"Successful configuration on device: {successful_device}")
    else:
        print("No successful camera configuration")
        sys.exit()
        
    list_cameras()
    
    
