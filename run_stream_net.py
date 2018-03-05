import tensorflow as tf
import sys
import os
import numpy as np
from stable_graph_feeder import WholeGraph as WG
from make_sets import Setmaker as SM
RunGraph = WG()
SetMaker = SM()

file = "dataTEST/inhale/1.wav"
data = SetMaker.load_blind(file)
prediction = RunGraph.make_prediction(data)
print(prediction)