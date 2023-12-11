import subprocess
import os
import time
import pyaudio
import numpy as np
import concurrent.futures

class VideoRecorder:
    def __init__(self):
        self.video_filename = "output.h264"
        self.sound_ts = None

    def record_video(self):
        try:
            # Record video using libcamera-vid with circular buffer
            # --width 1152 --height 192 --denoise cdn_off --framerate 304 --shutter 1000 -o {self.video_filename} -n
            record_cmd = f"libcamera-vid --circular 1 --inline --width 1152 --height 192 --denoise cdn_off --framerate 304 --shutter 500 -t 30000 -o {self.video_filename} -n"
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

    def convert_to_mp4(self):
            # Convert the cropped video to MP4 using ffmpeg
            mp4_cmd = f"ffmpeg -r 30 -y -i {self.video_filename} {self.video_filename}.mp4"
            subprocess.run(mp4_cmd, shell=True, check=True)

    def run(self):
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
                print(f"Error: {e}")
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

if __name__ == "__main__":
    video_recorder = VideoRecorder()
    video_recorder.run()
