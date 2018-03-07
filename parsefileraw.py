import wave
import sys
file = "streamtest/justalongthing.wav"
stream = wave.open(file, 'r')

for i in range(20):
    print(stream.readframes(1000))
