import pyaudio
import numpy as np

CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100  # Sample rate (samples per second)

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening for sound...")

try:
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        # Check if the data is not empty
        if len(data) > 0:
            # Calculate the root mean square (RMS) to detect sound
            rms = np.sqrt(np.mean(np.square(data)))
            if rms > 1000:  # Adjust this threshold based on your environment
                print("Sound detected!")
except KeyboardInterrupt:
    print("Stopped by user")

stream.stop_stream()
stream.close()
p.terminate()
