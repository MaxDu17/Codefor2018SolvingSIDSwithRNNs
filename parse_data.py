import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import struct
import wave
CHUNK = 4096
RECORDTIME = 2
test_fraction = 0.1
valid_fraction = 0.1
import random

class Source:
    class Native:
        INHALE_DIR = "dataSPLIT/inhale/"
        EXHALE_DIR = "datasplit/exhale/"
        UNKNOWN_DIR= "dataSPLIT/unknown/"
    class Server:
        INHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        EXHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        UNKNOWN_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"


def load_wav_file(name):
    wav_file = wave.open(name, 'r')
    data = wav_file.readframes(RECORDTIME*CHUNK)
    wav_file.close()
    data = struct.unpack('{n}h'.format(n=RECORDTIME*CHUNK), data)
    data = np.array(data)
    #data = (data+128)/255
    return data

file_name = Source.Native.INHALE_DIR + "1.wav"
test = load_wav_file(file_name)
file_name = Source.Native.EXHALE_DIR + "5.wav"
test1 = load_wav_file(file_name)
output = np.abs(np.fft.fft(test))
output = output[1:CHUNK]
output = (output - np.min(output))/(np.max(output)-np.min(output))
frequency_div = np.linspace(1,CHUNK/2, CHUNK-1)


output1 = np.abs(np.fft.fft(test1))
output1 = (output1 - np.min(output1))/(np.max(output1)-np.min(output1))
output1 = output1[1:CHUNK]


plt.plot(frequency_div, output,output1)
plt.show()