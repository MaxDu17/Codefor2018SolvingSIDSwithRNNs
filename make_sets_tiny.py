
import numpy as np
from parse_data import DataParse as dp

import random

class Source:
    class Current:
        INHALE_DIR = "/home/wedu/Desktop/VolatileRepos/finalcode/data_tiny/inhale/"
        EXHALE_DIR = "/home/wedu/Desktop/VolatileRepos/finalcode/data_tiny/exhale/"
        UNKNOWN_DIR = "/home/wedu/Desktop/VolatileRepos/finalcode/data_tiny/unknown/"
    class Native:
        INHALE_DIR = "data_tiny/inhale/"
        EXHALE_DIR = "data_tiny/exhale/"
        UNKNOWN_DIR= "data_tiny/unknown/"
    class Server:
        INHALE_DIR = "/home/wedu/Desktop/VolatileRepos/finalcode/data_tiny/inhale/"
        EXHALE_DIR = "/home/wedu/Desktop/VolatileRepos/finalcode/data_tiny/exhale/"
        UNKNOWN_DIR = "/home/wedu/Desktop/VolatileRepos/finalcode/data_tiny/unknown/"
class Setmaker:

    TESTFRACTION = 0.1
    VALIDFRACTION = 0.1
    TRAINFRACTION = 1
    TOTALPOINTS = 9

    file_maker = dp()
    exempt_set = list()  # this is for the test
    train_list = list()
    test_list = list()
    validation_list = list()

    def pick_train(self):
        big_set = list()

        for i in range(self.TOTALPOINTS):
            big_set.append(i)

        return big_set



    def get_batch_arrays(self):
        train = self.pick_train()
        return train


    def load_next_epoch(self): #more or less a wrapper function

        self.train_list=  self.get_batch_arrays()
        #self.set_validator(train=self.train_list,test=self.exempt_set,valid=self.validation_list) #this is to check the set


    def get_test_set(self): #call this before everything! It's a wrapper function!

        self.test_list = self.pick_test()
        print("creating test set!!")
        return self.test_list


    def one_hot_from_label(self,label):
        scratch_vector = np.zeros(3)
        if label == 'inhale':
            scratch_vector[0] = 1
            return scratch_vector
        elif label == 'exhale':
            scratch_vector[1] = 1
            return scratch_vector
        elif label == 'unknown':
            scratch_vector[2] = 1
            return scratch_vector
        else:
            print("not recognized! Traceback: raised from one_hot_from_label from make_sets.py")


    def next_test(self,batch_number):
        if batch_number > 239:
            print("you have exceeded the batch! Try again! This is the test round")
        batch_index = self.test_list[batch_number]
        if batch_index < 100:
            label = 'inhale'
            file_name = Source.Current.INHALE_DIR + str(batch_index) + ".wav"
            data_list = self.file_maker.prepare_data(file_name)
            return data_list, label

        elif batch_index >= 100 and batch_index < 200:
            label = 'exhale'
            file_name = Source.Current.EXHALE_DIR + str(batch_index - 100) + ".wav"
            data_list = self.file_maker.prepare_data(file_name)
            return data_list, label
        else:
            label = 'unknown'
            file_name = Source.Current.UNKNOWN_DIR + str(batch_index - 200) + ".wav"
            data_list = self.file_maker.prepare_data(file_name)
            return data_list, label

    def next_validation(self,batch_number):
        if batch_number > 239:
            print("you have exceeded the batch! Try again! This is the test round")
        batch_index = self.validation_list[batch_number]
        if batch_index < 100:
            label = 'inhale'
            file_name = Source.Current.INHALE_DIR + str(batch_index) + ".wav"
            data_list = self.file_maker.prepare_data(file_name)
            return data_list, label

        elif batch_index >= 100 and batch_index < 200:
            label = 'exhale'
            file_name = Source.Current.EXHALE_DIR + str(batch_index - 100) + ".wav"
            data_list = self.file_maker.prepare_data(file_name)
            return data_list, label
        else:
            label = 'unknown'
            file_name = Source.Current.UNKNOWN_DIR + str(batch_index - 200) + ".wav"
            data_list = self.file_maker.prepare_data(file_name)
            return data_list, label

    def load_next_train_sample(self, batch_number):
       # print("you are on batch" , batch_number)
        if batch_number >8 or batch_number < 0:
            print("you have exceeded the batch! Try again!")
        batch_index = self.train_list[batch_number]
        if batch_index <3:
            label = 'inhale'
            file_name = Source.Current.INHALE_DIR + str(batch_index) + ".wav"
            data_list = self.file_maker.prepare_data(file_name)
            return data_list, label

        elif batch_index >=3 and batch_index < 6:
            label = 'exhale'
            file_name = Source.Current.EXHALE_DIR + str(batch_index-3) + ".wav"
            data_list = self.file_maker.prepare_data(file_name)
            return data_list, label
        else:
            label = 'unknown'
            file_name = Source.Current.UNKNOWN_DIR + str(batch_index - 6) + ".wav"
            data_list = self.file_maker.prepare_data(file_name)
            return data_list, label

    def set_validator(self,train,valid,test): #checks if everything is there
        for i in range(300):
            try:
                traintest = train.index(i)
                print('train ', i)
            except:
                try:
                    validtest = valid.index(i)
                    print('valid ', i)
                except:
                    try:
                        testtest = test.index(i)
                        print('test ', i)
                    except:
                        print("Oops we have an error")
        print("all good!")

def test_library():
    maker = Setmaker()
    print("this is the test set assignment: " ,maker.get_test_set())
    maker.load_next_epoch()
    test,label_test= maker.load_next_train_sample(10)
    print("this is how many time frames: ",len(test))
    print("This is how many frequency bins: ",len(test[0]))
    print("this is the label for this set: ", label_test)
    print("and this is the raw data: ", test)
    valid, label_valid = maker.next_validation(10)
    print("This is how long the test set is: ",len(maker.exempt_set))
    print("This is how long the train set is: ",len(maker.train_list))
    print("This is how long the validation set is: ", len(maker.validation_list))
    print("This is what the validation label is: ",label_valid)

#test_library()