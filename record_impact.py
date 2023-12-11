import subprocess
import os
import time
import pyaudio
import numpy as np

class VideoRecorder:
    def __init__(self):
        self.video_filename = "output.h264"
        self.sound_ts = None

    def record_video(self):
        try:
            # Record video using libcamera-vid with circular buffer
            # --width 1152 --height 192 --denoise cdn_off --framerate 304 --shutter 1000 -o {self.video_filename} -n
            record_cmd = f"libcamera-vid --circular 3 --inline --width 1152 --height 192 --denoise cdn_off --framerate 304 --shutter 1000 -o {self.video_filename} -n"
            subprocess.run(record_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during video recording: {e}")

    def wait_for_hit2(self, stream, chunk_size=1024, threshold=1000):
        while True:
            data = stream.read(chunk_size)
            audio_signal = np.frombuffer(data, dtype=np.int16)

            # Check if the amplitude exceeds the threshold, stop video
            if np.max(np.abs(audio_signal)) > threshold:
                print("Click detected! Triggering action.")
                subprocess.run("pkill -SIGINT libcamera-vid", shell=True)
                self.sound_ts = time.time()
                break
            else:
                print("no sound", end='\r')
        
    def wait_for_hit(self):
        # Simulate waiting for a click sound (replace this with your actual mechanism)
        time.sleep(5)  # Placeholder for waiting for a hit

        # Signal libcamera-vid to exit
        subprocess.run("pkill -SIGINT libcamera-vid", shell=True)

        # Record the timestamp of the sound
        self.sound_ts = time.time()

    def crop_video(self):
        if self.sound_ts:
            # Crop the video using ffmpeg based on the sound timestamp
            crop_cmd = f"ffmpeg -y -ss {self.sound_ts} -i {self.video_filename} -t 2 -c copy cropped_{self.video_filename}"
            subprocess.run(crop_cmd, shell=True, check=True)
        else:
            print("Cannot crop video: No sound timestamp available.")

    def convert_to_mp4(self):
        # Convert the cropped video to MP4 using ffmpeg
        mp4_cmd = f"ffmpeg -y -i {self.video_filename} -c:v copy -c:a aac -strict experimental output.mp4"
        subprocess.run(mp4_cmd, shell=True, check=True)

    def run(self):
        # Record video
        self.record_video()

        # Wait for hit sound
        #self.wait_for_hit()
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024
            )
            self.wait_for_hit2()
        except Exception as e:
            print(f"Error: {e}")
            # Handle the exception as needed
        finally:
            # Perform any cleanup or resource release here
            if 'stream' in locals() and stream is not None:
                stream.stop_stream()
                stream.close()
            if 'p' in locals() and p is not None:
                p.terminate()

        # Crop video based on sound timestamp
        #self.crop_video()

        # Convert to MP4
        self.convert_to_mp4()


if __name__ == "__main__":
    video_recorder = VideoRecorder()
    video_recorder.run()
