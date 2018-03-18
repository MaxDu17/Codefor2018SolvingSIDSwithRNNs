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

TOTALPOINTS = 200
TOTALINCLUSION = 50
SKIP = 24576
CHUNK = 4096
RECORDTIME = 2
total_list = list()
big_list = list()
#big_list = [0]*819200

class Source:
    class Current:
        INHALE_DIR = "sen_data/inhale/"
        EXHALE_DIR = "sen_data/exhale/"
        UNKNOWN_DIR= "dataSPLIT/unknown/"
    class Native:
        INHALE_DIR = "sen_data/inhale/"
        EXHALE_DIR = "sen_data/exhale/"
        UNKNOWN_DIR= "sen_data/unknown/"
    class Server:
        INHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/sen_data/inhale/"
        EXHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/sen_data/exhale/"
        UNKNOWN_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/sen_data/unknown/"

def create_beginning():
    global big_list
    wav_file = wave.open("streamtest/ambience.wav", 'r')
    data = wav_file.readframes(819200)
    print(len(data))
    data = struct.unpack('{n}h'.format(n=819200), data)
    print(len(data))
    big_list[0:0] = data
    print(len(big_list))
def create_set():
    create_beginning()
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
            parse_iter = ['inhale',((k+1)*SKIP)-8192,(((k+1)*SKIP))/4096]
            writer.writerow(parse_iter)
        elif audio_select_list[k] >=100 and audio_select_list[k] <200:
            open_name = Source.Current.EXHALE_DIR + str(k) + ".wav"
            parse_iter = ['exhale',((k+1)*SKIP)-8192,(((k+1)*SKIP))/4096]
            writer.writerow(parse_iter)
        else:
            print("insert unknown")
        wav_file = wave.open(open_name, 'r')
        data = wav_file.readframes(RECORDTIME * CHUNK)
        wav_file.close()

        data = struct.unpack('{n}h'.format(n=RECORDTIME * CHUNK), data)
        data = np.array(data)
        big_list[(k+1)*SKIP:(k+1)*SKIP] = data


    print(len(big_list))

    wf = wave.open("streamtest/five_minutes.wav", 'wb')

    wf.setparams(wav_file.getparams())
    print(big_list)
    for sample in big_list:
        value = struct.pack('h', sample)
        wf.writeframesraw(value)

    wf.close()

def create_blank():
    wav_file = wave.open(Source.Current.UNKNOWN_DIR + str(5) + ".wav", 'r')
    wf = wave.open("streamtest/5minsblank.wav", 'wb')
    big_list = [0] * 1228800
    wf.setparams(wav_file.getparams())
    for sample in big_list:
        value = struct.pack('h', sample)
        wf.writeframesraw(value)

    wf.close()

create_set()