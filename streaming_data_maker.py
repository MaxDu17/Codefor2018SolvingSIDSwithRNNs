import numpy as np
import wave
import random
import csv
import struct
import pyaudio
import sys


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 4096

TOTALPOINTS = 300
TOTALINCLUSION = 50
SKIP = 24576
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

try:
    f = open('streamtest/ground_truth.csv','w')
except:
    print("pls close files")
    sys.exit()
writer = csv.writer(f,lineterminator="\n")



for k in range(TOTALINCLUSION):
    if audio_select_list[k] >299:
        raise Exception("Out of bounds! Sorry!")
    if audio_select_list[k] <100:
        open_name = Source.Current.INHALE_DIR + str(k) + ".wav"
        parse_iter = ['inhale',((k+1)*SKIP)-8192,(((k+1)*SKIP)-8192)/4096]
        writer.writerow(parse_iter)
    elif audio_select_list[k] >=100 and audio_select_list[k] <200:
        open_name = Source.Current.EXHALE_DIR + str(k) + ".wav"
        parse_iter = ['exhale',((k+1)*SKIP)-8192,(((k+1)*SKIP)-8192)/4096]
        writer.writerow(parse_iter)
    else:
        open_name = Source.Current.UNKNOWN_DIR + str(k) + ".wav"
        parse_iter = ['unknown',((k+1)*SKIP)-8192,(((k+1)*SKIP)-8192)/4096]
        writer.writerow(parse_iter)

    wav_file = wave.open(open_name, 'r')
    data = wav_file.readframes(RECORDTIME * CHUNK)
    wav_file.close()

    data = struct.unpack('{n}h'.format(n=RECORDTIME * CHUNK), data)
    data = np.array(data)
    big_list[(k+1)*SKIP:(k+1)*SKIP] = data


print(len(big_list))

wf = wave.open("streamtest/5minsampl.wav", 'wb')

wf.setparams(wav_file.getparams())
for sample in big_list:
    value = struct.pack('h', sample)
    wf.writeframesraw(value)

wf.close()

