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


def split_file(data):

    time_split = list()
    for i in range(16):
        current_block_start = i*512
        current_block_end = current_block_start + 512
        time_split.append(data[current_block_start:current_block_end])

    return time_split


def load_fourier(data):

    fourier_output = np.abs(np.fft.fft(data))
    fourier_output = fourier_output[1:int(CHUNK/16)]#truncates according to nyquist limit
    fourier_output_norm = (fourier_output - np.min(fourier_output)) / (np.max(fourier_output) - np.min(fourier_output))
    frequency_division = np.linspace(1,CHUNK/2, int((CHUNK/16)-1))
    return fourier_output_norm, frequency_division

def bin(output):
    output_bins = [None]*43

    for i in range(42):
        first = i*6
        last = first+6
        output_bins[i] = np.mean(output[first:last])
    output_bins[42] = np.mean(output[42*6:])
    return output_bins

def prepare_data(name):
    data_list = list()
    data = load_wav_file(name)
    time_split = split_file(data)
    for time_slice in time_split:
        output, frq_div = load_fourier(time_slice)
        output_bins = bin(output)
        data_list.append(output_bins)
    return data_list


file_name = Source.Native.EXHALE_DIR + "1.wav"
data_list = prepare_data(file_name)
print(len(data_list))
'''
output1 = np.abs(np.fft.fft(test1))
output1 = (output1 - np.min(output1))/(np.max(output1)-np.min(output1))
output1 = output1[1:CHUNK]


plt.plot(frq_div, output)
plt.show()'''