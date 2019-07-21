from stable_graph_feederv6 import WholeGraph as WG
import csv
import numpy as np
WholeGraph = WG()
prediction_dictionary = {0:"inhale", 1:"exhale", 2:"unknown"}
from parse_data import DataParse as dp
class Source:
    class Native:
        INHALE_DIR = "sen_data/inhale/"
        EXHALE_DIR = "sen_data/exhale/"
        UNKNOWN_DIR= "sen_data/unknown/"
    class Server:
        INHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        EXHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        UNKNOWN_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"

dataprocess = dp()
k = open("debugging/v6check.csv", "w")
writer_log = csv.writer(k, lineterminator="\n")
for i in range(200):
    name = Source.Native.INHALE_DIR + str(i) + ".wav"

    data = dataprocess.load_wav_file(name)

    prediction = WholeGraph.make_prediction(data)
    if(np.argmax(prediction) != 0):
        carrier = [i,"inhale", prediction_dictionary[np.argmax(prediction)], prediction]
        writer_log.writerow(carrier)
    print(i)


for i in range(400):
    name = Source.Native.UNKNOWN_DIR + str(i) + ".wav"

    data = dataprocess.load_wav_file(name)
    prediction = WholeGraph.make_prediction(data)
    if(np.argmax(prediction) != 2):
        carrier = [i,"unknown", prediction_dictionary[np.argmax(prediction)], prediction]
        writer_log.writerow(carrier)
    print(i)


for i in range(200):
    name = Source.Native.EXHALE_DIR + str(i) + ".wav"
    data = dataprocess.load_wav_file(name)
    prediction = WholeGraph.make_prediction(data)
    if(np.argmax(prediction) != 1):
        carrier = [i,"exhale", prediction_dictionary[np.argmax(prediction)], prediction]
        writer_log.writerow(carrier)
    print(i)



print("done!")

