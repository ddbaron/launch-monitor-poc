import subprocess
import os
import time
import sys
import pyaudio
import numpy as np
import concurrent.futures

class VideoRecorder:
    def __init__(self):
        self.video_filename = "/dev/shm/output.h264"
        self.save_pts = "/dev/shm/tst.pts"
        self.sound_ts = None
        ### 1440x480@132, 1440x320@193, 1200x208@283, 1152x192x304, 672x128@427, 816x144@387
        self.width = 1440
        self.height = 320
        self.fps = 193
        self.shutter = 250
        self.threshold = 10000 # sound threshold to hear impact


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
            # --width 1152 --height 192 --denoise cdn_off --framerate 304 --shutter 1000 -o {self.video_filename} -n
            record_cmd = f"libcamera-vid --level 4.2 --circular 1 --inline --width {self.width} --height {self.height} --framerate {self.fps} --shutter {self.shutter} --denoise cdn_off --save-pts {self.save_pts} -t 0 -o {self.video_filename} -n"
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

    def run(self):

        # Configure media-ctrl
        successful_device = self.media_ctl()
        if successful_device:
            print(f"Successful configuration on device: {successful_device}")
        else:
            print("No successful configuration")
            sys.exit()

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

        # Continue with other processing after both tasks are complete
        print("Video recording and sound detection completed. Converting to mp4 video.")
        self.convert_to_mp4()

        # Run ptsanalyze to determine fps and frameskips
        self.ptsanalyze()

if __name__ == "__main__":
    video_recorder = VideoRecorder()
    video_recorder.run()
