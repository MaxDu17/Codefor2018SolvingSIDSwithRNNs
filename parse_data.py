import tensorflow as tf
import numpy as np

import pyaudio
import wave
CHUNK = 4096
test_fraction = 0.1
valid_fraction = 0.1
import random

class Source:
    class Native:
        INHALE_DIR = "dataSPLIT/inhale/"
        EXHALE_DIR = "C:/Users/wedu/Desktop/Working Repository/finalcode/dataSPLIT/exhale/"
        UNKNOWN_DIR= "dataSPLIT/unknown/"
    class Server:
        INHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        EXHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        UNKNOWN_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"


def load_wav_file(name):
    file_object = wave.open(name, "rb")
    print("I am loading" + name)
    chunk = []
    _data = file_object.readframes(CHUNK)
    while _data:
        data = np.fromstring(_data, dtype = 'uint8')
        data = (data + 128)/255
        chunk.extend(data)
        _data = file_object.readframes(CHUNK)
    chunk = chunk[0:CHUNK*2]
    chunk.extend(np.zeros(CHUNK*2-len(chunk)))
    return chunk

file_name = Source.Server.EXHALE_DIR + "0"
test = load_wav_file(file_name)
print(test)