import subprocess
import cv2
import os
import time
import sys
import pyaudio
import argparse
import numpy as np
import concurrent.futures

class VideoRecorder:
    def __init__(self):
        self.video_filename = "/dev/shm/output.h264"
        self.save_pts = "/dev/shm/tst.pts"
        ### libcamera-vid and media-ctl settings
        ## 1440x480@132, 1440x320@193, 1200x208@283, 1152x192x304, 816x144@387, 672x128@427
        self.width = 1440
        self.height = 480
        self.fps = 132
        self.shutter = 250
        self.gain = 16.0 # 1.0 to 16.0; no default
        self.brightness = 0.25 # -1.0 to 1.0; 0.0 is normal
        self.contrast = 1.0 # 0.0 to 32.0
        self.saturation = 1.0 # 0.0 to 32.0 (0.0 greyscale, 1.0 is normal)
        self.sharpness = 1.0 # 0.0 to 16.0; 1.0 is normal
        ## audio ##
        self.threshold = 10000 # sound threshold to hear impact
        self.sound_ts = None
        ## camera sensor ##
        self.focal_length_mm = 6  # 6mm lens
        self.pixel_size_um = 3.45  # Pixel size in µm
        self.sensor_diagonal_mm = 6.3  # Sensor diagonal in mm
        self.resolution = (self.width, self.height)  # Sensor resolution

    def media_ctl(self):
         # Loop through media devices and set V4L2 format
        offset_x = (1456 - self.width) // 2
        offset_y = (1088 - self.height) // 2

        print(f"calling media-ctl")

        for m in range(0, 6):
            try:
                command = f'media-ctl -d /dev/media{m} --set-v4l2 "\'imx296 10-001a\':0 [fmt:SBGGR10_1X10/{self.width}x{self.height} crop:({offset_x},{offset_y})/{self.width}x{self.height}]"'
                print(f"Command: {command}")
                subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                return f'/dev/media{m}'
            except subprocess.CalledProcessError as e:
                print(f"Error configuring media-ctl: {e}")
        return None
    
    def record_video(self):
        try:
            # Record video using libcamera-vid with circular buffer
            record_cmd = f"libcamera-vid --level 4.2 --circular 1 --inline --width {self.width} --height {self.height} \
                --framerate {self.fps} --shutter {self.shutter} \
                --gain {self.gain} --brightness {self.brightness} --contrast {self.contrast} \
                --saturation {self.saturation} --sharpness {self.sharpness} \
                --denoise cdn_off --save-pts {self.save_pts} -t 0 -o {self.video_filename} -n"
            print(f"libcamera-vid config: {record_cmd}.")
            subprocess.run(record_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during video recording: {e}")

    def wait_for_hit2(self, stream, chunk_size=1024, threshold=5000):
        while True:
            data = stream.read(chunk_size)
            audio_signal = np.frombuffer(data, dtype=np.int16)

            # Check if the amplitude exceeds the threshold, stop video
            if np.max(np.abs(audio_signal)) > self.threshold:
                print("Click detected! Triggering action.")
                subprocess.run("pkill -SIGINT libcamera-vid", shell=True)
                self.sound_ts = time.time()
                break
            else:
                print("Waiting for sound...", end='\r')

    def convert_to_mp4(self):
            # Convert the cropped video to MP4 using ffmpeg
            mp4_cmd = f"ffmpeg -y -r {int(self.fps) / 10} -i {self.video_filename} impact.mp4"
            subprocess.run(mp4_cmd, shell=True, check=True)

    def estimate_focal_length_in_pixels(self, focal_length_mm, pixel_size_um, sensor_diagonal_mm, resolution):
        """
        Estimate the focal length in pixels for a given camera setup.

        :param focal_length_mm: Focal length in millimeters (mm).
        :param pixel_size_um: Pixel size in micrometers (µm).
        :param sensor_diagonal_mm: Sensor diagonal size in millimeters (mm).
        :param resolution: Resolution of the sensor as a tuple (width, height).
        :return: Estimated focal length in pixels.
        """
        # Convert pixel size to millimeters
        pixel_size_mm = pixel_size_um / 1000

        # Calculate sensor width and height based on diagonal size
        # Assuming a 4:3 aspect ratio for 1/2.9" sensor
        aspect_ratio = 4 / 3
        sensor_height_mm = (sensor_diagonal_mm / ((1 + aspect_ratio ** 2) ** 0.5))
        sensor_width_mm = sensor_height_mm * aspect_ratio

        # Calculate the pixel density (pixels per mm)
        pixel_density_width = resolution[0] / sensor_width_mm
        pixel_density_height = resolution[1] / sensor_height_mm

        # Calculate focal length in pixels
        focal_length_px_width = focal_length_mm * pixel_density_width
        focal_length_px_height = focal_length_mm * pixel_density_height

        return focal_length_px_width, focal_length_px_height

    def cv_overlay_mp4(self):
        # Open the video file
        cap = cv2.VideoCapture(self.video_filename)

        # Check if the video file was opened successfully
        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()

        # Get the width and height of the video frames
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Extract dir, filename and extension
        directory_path = os.path.dirname(self.video_filename)
        filename, extension = os.path.splitext(os.path.basename(self.video_filename))

        # Create a VideoWriter object with the same filename but with 'CV'
        output_filename = f'rec_impact3_{filename}_{width}x{height}.mp4'
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), int(self.fps)/10, (width, height))

        # Known real-world diameter of the golf ball in millimeters
        real_world_diameter_mm = 42.67

        # Estimated focal length in pixels (TODO: this value needs to be obtained through camera calibration)
        focal_length_px = video_recorder.estimate_focal_length_in_pixels(self.focal_length_mm, self.pixel_size_um, self.sensor_diagonal_mm, self.resolution)
        print(f"Estimated focal Length in Pixels (Width, Height): {focal_length_px}")
        focal_length = focal_length_px[0]  # Estimated using width dimension

        # Initialize counters
        total_frames = 0
        golf_ball_detected_frames = 0

        while True:
            # Read the next frame
            ret, frame = cap.read()

            # Increment total frame count
            total_frames += 1

            # Break the loop if the video has ended
            if not ret:
                print("No more frames to read. Exiting.")
                break

            # Resize the frame
            resized_frame = cv2.resize(frame, (width, height))

            # Color-based thresholding for the golf ball
            lower_color = np.array([20, 100, 100])
            upper_color = np.array([40, 255, 255])
            hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, lower_color, upper_color)

            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            golf_ball_detected = False  # Flag to check if golf ball is detected in this frame

            # Object detection
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    golf_ball_detected = True
                    golf_ball_detected_frames += 1

                    # Fit a circle around the contour
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))

                    # Draw the circle and dot
                    cv2.circle(resized_frame, center, int(radius), (0, 255, 0), 2)
                    cv2.circle(resized_frame, center, 5, (0, 0, 255), -1)

                    # Calculate the depth (z-coordinate)
                    depth = (focal_length * real_world_diameter_mm) / (2 * radius)

                    # Display XYZ coordinates
                    coordinates_text = "XYZ: ({:.2f}, {:.2f}, {:.2f})".format(center[0], center[1], depth)
                    cv2.putText(resized_frame, coordinates_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Output a message with real-world coordinates
                    print("Golf ball detected at real-world coordinates (x, y, z): ({:.2f}, {:.2f}, {:.2f})".format(center[0], center[1], depth))

            # Detection status for this frame
            if not golf_ball_detected:
                print(f"Frame {total_frames}: Golf ball not detected.")

            # Write the processed frame to the output video
            out.write(resized_frame)

        # After the loop, print summary statistics
        print(f"Total number of frames read: {total_frames}")
        print(f"Number of frames where golf ball was detected: {golf_ball_detected_frames}")

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def ptsanalyze(self):
        try:
            # Remove existing timestamp analysis file
            tstamps_cmd = f"rm -f tstamps.csv"
            subprocess.run(tstamps_cmd, shell=True, check=True)

            # Run the ptsanalyze bash script with the specified input file
            pts_cmd = f"ptsanalyze {self.save_pts}"
            subprocess.run(pts_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during ptsanalyze: {e}")

    def display_fov(self):

        self.media_ctl()

        try:
            # show window to enable user to place ball within window
            record_cmd = f"libcamera-vid --width {self.width} --height {self.height} -t 0"
            print(f"libcamera-vid display_fov config: {record_cmd}.")
            subprocess.run(record_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during display_fov: {e}")

    def run(self):

        # Configure media-ctrl
        successful_device = self.media_ctl()
        if successful_device:
            print(f"Successful configuration on device: {successful_device}")
        else:
            print("No successful configuration")
            sys.exit()

        # Record video until impact is heard, then save off the last 1 second
        # Use ThreadPoolExecutor to run record_video() concurrently with wait_for_hit2()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start recording video concurrently
            video_future = executor.submit(self.record_video)

            try:
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024
                )
                # Wait for hit sound concurrently
                audio_future = executor.submit(self.wait_for_hit2, stream)
                audio_future.result()  # Wait for audio_future to complete

            except Exception as e:
                print(f"Audio Error: {e}")
                # Handle the exception as needed
            finally:
                # Perform any cleanup or resource release here
                if 'stream' in locals() and stream is not None:
                    stream.stop_stream()
                    stream.close()
                if 'p' in locals() and p is not None:
                    p.terminate()

            # Wait for video recording to complete
            video_future.result()

        # Begin post processing
        print("Video recording and sound detection completed. Begining post processing.")
        
        #self.convert_to_mp4()

        # read file with CV, detect ball and xyz and write out mp4 with CV overlays
        self.cv_overlay_mp4()

        # Run ptsanalyze on the .h264 to determine fps and frameskips
        self.ptsanalyze()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Monitor")
    parser.add_argument("--fov", type=int, default=0, help="Display the FOV so the ball can be oriented manually")
    args = parser.parse_args()

    video_recorder = VideoRecorder()

    if args.fov == 1:
        video_recorder.display_fov()        
    else:
        video_recorder.run()