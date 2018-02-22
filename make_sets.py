import tensorflow as tf
import numpy as np
from parse_data import DataParse as dp

import random

TESTFRACTION = 0.1
VALIDFRACTION = 0.1
TRAINFRACTION = 0.8
TOTALPOINTS = 300

train_list = list()
test_list = list()
validation_list = list()


class Source:
    class Native:
        INHALE_DIR = "dataSPLIT/inhale/"
        EXHALE_DIR = "datasplit/exhale/"
        UNKNOWN_DIR= "dataSPLIT/unknown/"
    class Server:
        INHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        EXHALE_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"
        UNKNOWN_DIR = "/home/wedu/Desktop/VolatileRepos/DatasetMaker/dataSPLIT/inhale/"

def pick_train():
    big_set = list()
    real_set = list()
    for i in range(TOTALPOINTS):
        big_set.append(i)
    real_set = random.sample(big_set,int(TOTALPOINTS*TRAINFRACTION))
    return real_set


def pick_test(train_set):
    big_set = list()
    real_set = list()
    for i in range(TOTALPOINTS):
        big_set.append(i)
    leftover_set = [k for k in big_set if k not in train_set]
    real_set = random.sample(leftover_set,int(TOTALPOINTS*VALIDFRACTION))
    return real_set

def pick_valid(train_set, test_set ):
    big_set = list()
    real_set = list()
    for i in range(TOTALPOINTS):
        big_set.append(i)
    leftover_set = [k for k in big_set if k not in train_set and k not in test_set]
    return leftover_set

def get_batch_arrays():
    train = pick_train()
    test = pick_test(train)
    validation = pick_valid(train, test)
    return train, test, validation

def load_next_epoch(): #more or less a wrapper function
    global train_list
    global test_list
    global validation_list
    train_list, test_list, validation_list = get_batch_arrays()

def load_next_train_batch(batch_number):
    global train_list
    batch_index = train_list[batch_number]
    