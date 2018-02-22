import tensorflow as tf
import numpy as np
from parse_data import DataParse as dp

import random

TESTFRACTION = 0.1
VALIDFRACTION = 0.1
TRAINFRACTION = 0.9
TOTALPOINTS = 300

file_maker = dp()
exempt_set = list() #this is for the test
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
class Setmaker:
    def pick_train(self):
        big_set = list()
        real_set = list()
        global exempt_set

        for i in range(TOTALPOINTS):
            big_set.append(i)
        leftover_set = [k for k in big_set if k not in exempt_set]
        real_set = random.sample(leftover_set,int(TOTALPOINTS*TRAINFRACTION))
        return real_set


    def pick_valid(self,train_set):
        big_set = list()
        real_set = list()
        global exempt_set

        for i in range(TOTALPOINTS):
            big_set.append(i)
        leftover_set = [k for k in big_set if k not in train_set and k not in exempt_set]

        return leftover_set

    def pick_test(self):
        big_set = list()
        real_set = list()
        for i in range(TOTALPOINTS):
            big_set.append(i)
        test_set = random.sample(big_set,int(TOTALPOINTS*TESTFRACTION))
        exempt_set = test_set
        return test_set #returns test set


    def get_batch_arrays(self):
        train = self.pick_train()
        validation = self.pick_valid(train)
        return train, validation

    def load_next_epoch(self): #more or less a wrapper function
        global train_list
        global validation_list
        train_list, validation_list = self.get_batch_arrays()

    def get_test_set(self): #call this before everything! It's a wrapper function!
        global test_list
        test_list = self.pick_test()
        print("creating test set!!")
        return test_list

    def next_test(self,batch_number):
        if batch_number > 239:
            print("you have exceeded the batch! Try again! This is the test round")
        batch_index = train_list[batch_number]
        if batch_index < 100:
            label = 'inhale'
            file_name = Source.Native.INHALE_DIR + str(batch_index) + ".wav"
            data_list = file_maker.prepare_data(file_name)
            return data_list, label

        elif batch_index >= 100 and batch_index < 200:
            label = 'exhale'
            file_name = Source.Native.EXHALE_DIR + str(batch_index - 100) + ".wav"
            data_list = file_maker.prepare_data(file_name)
            return data_list, label
        else:
            label = 'unknown'
            file_name = Source.Native.UNKNOWN_DIR + str(batch_index - 200) + ".wav"
            data_list = file_maker.prepare_data(file_name)
            return data_list, label


    def load_next_train_batch(self, batch_number):
        global train_list
        global file_maker
        print("you are on batch" , batch_number)
        if batch_number >239:
            print("you have exceeded the batch! Try again!")
        batch_index = train_list[batch_number]
        if batch_index <100:
            label = 'inhale'
            file_name = Source.Native.INHALE_DIR + str(batch_index) + ".wav"
            data_list = file_maker.prepare_data(file_name)
            return data_list, label

        elif batch_index >=100 and batch_index < 200:
            label = 'exhale'
            file_name = Source.Native.EXHALE_DIR + str(batch_index-100) + ".wav"
            data_list = file_maker.prepare_data(file_name)
            return data_list, label
        else:
            label = 'unknown'
            file_name = Source.Native.UNKNOWN_DIR + str(batch_index - 200) + ".wav"
            data_list = file_maker.prepare_data(file_name)
            return data_list, label

maker = Setmaker()
print(maker.get_test_set())
maker.load_next_epoch()
test,label_test= maker.load_next_train_batch(10)
print(len(test))
print(len(test[0]))
print(label_test)
print(test)

print(len(exempt_set))
print(len(train_list))
print(len(validation_list))