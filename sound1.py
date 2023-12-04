import pyaudio
import numpy as np

def detect_click(stream, chunk_size=1024, threshold=1000):
    while True:
        data = stream.read(chunk_size)
        audio_signal = np.frombuffer(data, dtype=np.int16)

        # Example: Check if the amplitude exceeds the threshold
        if np.max(np.abs(audio_signal)) > threshold:
            print("Click detected! Triggering action.")
            break
        else:
            print("no sound", end='\r')

if __name__ == "__main__":
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024)

    detect_click(stream)
