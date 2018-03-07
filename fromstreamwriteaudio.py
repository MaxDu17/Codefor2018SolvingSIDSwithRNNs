import pyaudio
import sys
import wave
import keyboard
import time

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 4096
RECORD_SECONDS = 20

inhaleCounter = 0
exhaleCounter = 0
unknownCounter = 0
Time = time.clock()
life = True



    
def makeFile(stream, recorder,file_name,frames):
    stream.stop_stream()
    stream.close()
    recorder.terminate()

    wf = wave.open(file_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(recorder.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


def record():
    global inhaleCounter
    global exhaleCounter
    global unknownCounter
    global life
    Time = time.clock()
    recorder = pyaudio.PyAudio()
    stream = recorder.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
    
    frames = []
    while Time + RECORD_SECONDS > time.clock():
        data = stream.read(CHUNK)
        frames.append(data)
        print(data)
    makeFile(stream,recorder, "streamtest/justalongthing.wav",frames)



record()