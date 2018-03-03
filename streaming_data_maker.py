import numpy as np
import wave
import random
import csv
import struct
import pyaudio



FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 4096

TOTALPOINTS = 300
TOTALINCLUSION = 50
SKIP = 16384
CHUNK = 4096
RECORDTIME = 2
total_list = list()
big_list = [0]*819200

class Source:
    class Current:
        INHALE_DIR = "dataSPLIT/inhale/"
        EXHALE_DIR = "dataSPLIT/exhale/"
        UNKNOWN_DIR= "dataSPLIT/unknown/"
    class Native:
        INHALE_DIR = "dataSPLIT/inhale/"
        EXHALE_DIR = "dataSPLIT/exhale/"
        UNKNOWN_DIR= "dataSPLIT/unknown/"
    class Server:
        INHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        EXHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/exhale/"
        UNKNOWN_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/unknown/"



for i in range(TOTALPOINTS):
    total_list.append(i)

audio_select_list = random.sample(total_list, TOTALINCLUSION)
f = open('streamtest/ground_truth.csv','w')
writer = csv.writer(f,lineterminator="\n")



for k in range(TOTALINCLUSION):
    if audio_select_list[k] >299:
        raise Exception("Out of bounds! Sorry!")
    if audio_select_list[k] <100:
        open_name = Source.Current.INHALE_DIR + str(k) + ".wav"
        parse_iter = ['inhale',k*SKIP]
        writer.writerow(parse_iter)
    elif audio_select_list[k] >=100 and audio_select_list[k] <200:
        open_name = Source.Current.EXHALE_DIR + str(k) + ".wav"
        parse_iter = ['exhale',k*SKIP]
        writer.writerow(parse_iter)
    else:
        open_name = Source.Current.UNKNOWN_DIR + str(k) + ".wav"
        parse_iter = ['unknown',k*SKIP]
        writer.writerow(parse_iter)

    wav_file = wave.open(open_name, 'r')
    data = wav_file.readframes(RECORDTIME * CHUNK)
    wav_file.close()

    data = struct.unpack('{n}h'.format(n=RECORDTIME * CHUNK), data)
    data = np.array(data)
    big_list[k*SKIP:k*SKIP] = data
print(len(big_list))

wf = wave.open("streamtest/5minsampl.wav", 'wb')

wf.setparams(wav_file.getparams())
for sample in big_list:
    value = struct.pack('h', sample)
    wf.writeframesraw(value)

wf.close()

