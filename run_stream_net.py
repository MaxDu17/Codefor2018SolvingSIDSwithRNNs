import time
import numpy as np
import wave
import struct
from stable_graph_feederv5 import WholeGraph as WG
from make_sets import Setmaker as SM
from parse_data import DataParse as DP
from postprocesslen import Processor
import csv
import pyaudio
from significancetest import Trendtest as Tt
sigtest = Tt()

k = open("streamtest/real_time.csv", "w")
writer_log = csv.writer(k, lineterminator="\n")
p = open("streamtest/real_time_report_silent.csv", "w")
writer_log_raw = csv.writer(p, lineterminator="\n")

filter = Processor(file_name = "streamtest/predict_sig.csv")
ParseData = DP()
RunGraph = WG()
SetMaker = SM()
JUMPTIME = 2
FRAMERATE = 4096
CHUNK = 8192
OFFSET = 512
TIMEOUT = 4096
TIMEOUTSECS = TIMEOUT/FRAMERATE
ALPHALEVEL = 0.995
STREAMING_ALPHA_LEVEL = 0.90
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
    counter = 0
    file = "streamtest/five_minutes_test.wav"
    f = open("streamtest/peaks.csv","w")
    writer = csv.writer(f, lineterminator="\n")
    wav_file = wave.open(file, 'r')
    for i in range(2384):
        data = wav_file.readframes(CHUNK)
        wav_file.setpos((i+1)*OFFSET)
        data = struct.unpack('{n}h'.format(n=CHUNK), data)
        data = np.array(data)
        parsed_data = ParseData.bins_from_stream(data)
        prediction = RunGraph.make_prediction(parsed_data)
        writer.writerow(prediction[0])
        status = filter.process_data(prediction[0])
        if status:
            counter += 1
        if i%120 == 0:
            print(i/8)
            #print(counter)
            #sigtest.significance(counter)
            counter = 0
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
    filter.process_data(prediction[0])

def run_real_time():
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
#run_real_time()
test_from_file()
