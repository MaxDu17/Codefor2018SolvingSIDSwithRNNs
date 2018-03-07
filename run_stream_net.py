import tensorflow as tf
import sys
import time
import numpy as np
import wave
import struct
from stable_graph_feeder import WholeGraph as WG
from make_sets import Setmaker as SM
from parse_data import DataParse as DP
import csv
import pyaudio

k = open("streamtest/real_time.csv", "w")
writer_log = csv.writer(k, lineterminator="\n")
p = open("streamtest/real_time_report_silent.csv", "w")
writer_log_raw = csv.writer(p, lineterminator="\n")

ParseData = DP()
RunGraph = WG()
SetMaker = SM()
JUMPTIME = 2
FRAMERATE = 4096
CHUNK = 8192
OFFSET = 512
TIMEOUT = 2048
TIMEOUTSECS = TIMEOUT/FRAMERATE
ALPHALEVEL = 0.995
FORMAT = pyaudio.paInt16
timex = time.clock()
time_zero =time.clock()
prediction_dictionary = {0:"inhale", 1:"exhale", 2:"unknown"}
x = -1
last_x = -1
time_last = time.clock()
def test_implementation():
    file = "dataTEST/inhale/1.wav"
    data = SetMaker.load_blind(file)
    prediction = RunGraph.make_prediction(data)
    print(prediction)

def test_from_file():
    file = "streamtest/5minsampl.wav"
    f = open("streamtest/peaks.csv","w")
    k = open("streamtest/predictions.csv","w")
    writer_log = csv.writer(k,lineterminator="\n")
    writer = csv.writer(f, lineterminator="\n")
    wav_file = wave.open(file, 'r')
    x = -1
    last_x = -1
    time_diff = 99999
    for i in range(2399):
        data = wav_file.readframes(CHUNK)
        wav_file.setpos((i+1)*OFFSET)
        data = struct.unpack('{n}h'.format(n=CHUNK), data)
        data = np.array(data)
        parsed_data = ParseData.bins_from_stream(data)
        prediction = RunGraph.make_prediction(parsed_data)
        writer.writerow(prediction[0])
        x = np.argmax(prediction[0])
        time = i*OFFSET
        if x != 2:
            if prediction[0][x] > ALPHALEVEL:
                last_x = x
                time_last = i*OFFSET
        elif x == 2:
            if last_x !=2 and last_x != -1:
                if time-time_last>TIMEOUT:
                    carrier = [prediction_dictionary[last_x], i / 8]
                    writer_log.writerow(carrier)
                    last_x = x
            else:
                last_x = x
                pass
    wav_file.close()
def feed_and_output(data):
    global writer_log
    global writer_log_raw
    global last_x
    global x
    timex = time.clock()
    global time_last
    global time_zero
    prediction = RunGraph.make_prediction(data)
    x = np.argmax(prediction[0])
    writer_log_raw.writerow(prediction[0])
    if x != 2:
        if prediction[0][x] > ALPHALEVEL:
            last_x = x
            time_last = time.clock()
    elif x == 2:
        if last_x != 2 and last_x != -1:
            if timex - time_last > TIMEOUTSECS:
                carrier = [prediction_dictionary[last_x], timex-time_zero]
                writer_log.writerow(carrier)
                print(prediction_dictionary[last_x])
                last_x = x
        else:
            last_x = x
            pass
def real_time_now():
    recorder = pyaudio.PyAudio()
    stream = recorder.open(format = FORMAT, channels = 1, rate = FRAMERATE,input = True, frames_per_buffer = FRAMERATE)
    frames = []
    first = True
    data = []
    while True:
        if first:
            data = stream.read(CHUNK)
            data = struct.unpack('{n}h'.format(n=CHUNK), data)
            data=np.array(data)
            frames.extend(data)

            parsed_data = ParseData.bins_from_stream(frames)
            feed_and_output(parsed_data)
            first=False
        else:
            data = stream.read(OFFSET)
            data = struct.unpack('{n}h'.format(n=OFFSET), data)
            data = np.array(data)
            frames.extend(data)
            for i in range(OFFSET):
                del(frames[0])
            parsed_data = ParseData.bins_from_stream(frames)
            feed_and_output(parsed_data)

def emulate_stream():

    file = "streamtest/justalongthing.wav"
    stream = wave.open(file, 'r')
    frames = []
    first = True
    data = []
    while True:
        if first:
            data = stream.readframes(CHUNK)
            data = struct.unpack('{n}h'.format(n=CHUNK), data)
            data=np.array(data)
            frames.extend(data)

            parsed_data = ParseData.bins_from_stream(frames)
            feed_and_output(parsed_data)
            first=False
        else:
            data = stream.readframes(OFFSET)
            data = struct.unpack('{n}h'.format(n=OFFSET), data)
            data = np.array(data)
            frames.extend(data)
            for i in range(OFFSET):
                del(frames[0])
            parsed_data = ParseData.bins_from_stream(frames)
            feed_and_output(parsed_data)


emulate_stream()