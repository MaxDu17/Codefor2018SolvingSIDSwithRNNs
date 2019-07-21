import tensorflow as tf
import numpy as np
from parse_data import DataParse as dp
import wave
import random
TESTFRACTION = 0.1
VALIDFRACTION = 0.1
TRAINFRACTION = 0.8
TOTALPOINTS = 300
class Source:
    class Native:
        INHALE_DIR = "dataSPLIT/inhale/"
        EXHALE_DIR = "datasplit/exhale/"
        UNKNOWN_DIR= "dataSPLIT/unknown/"
    class Server:
        INHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        EXHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        UNKNOWN_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"

