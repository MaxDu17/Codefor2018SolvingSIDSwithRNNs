import tensorflow as tf
import sys
import os
import numpy as np
import wave
import struct
from stable_graph_feeder import WholeGraph as WG
from make_sets import Setmaker as SM
from parse_data import DataParse as DP
import csv

ParseData = DP()
RunGraph = WG()
SetMaker = SM()
JUMPTIME = 2
FRAMERATE = 4096
CHUNK = 8192
OFFSET = 512
prediction_dictionary = {0:"inhale", 1:"exhale", 2:"unknown"}
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
    for i in range(2399):
        data = wav_file.readframes(CHUNK)
        wav_file.setpos((i+1)*OFFSET)
        data = struct.unpack('{n}h'.format(n=CHUNK), data)
        data = np.array(data)
        parsed_data = ParseData.bins_from_stream(data)
        prediction = RunGraph.make_prediction(parsed_data)
        writer.writerow(prediction[0])
        x = np.argmax(prediction[0])
        if x != 2:
            if prediction[0][x] > 0.95:
                last_x = x
        elif x == 2:
            if last_x !=2:
                carrier = [prediction_dictionary[last_x], i / 8]
                writer_log.writerow(carrier)
                last_x = x



    wav_file.close()

test_from_file()