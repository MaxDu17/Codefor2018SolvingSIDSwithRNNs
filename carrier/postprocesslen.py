import numpy as np
import csv

class Processor:

    def __init__(self, file_name):
        #alpha = 98, length = 5
        p = open(file_name, "w")
        self.writer_object = csv.writer(p,lineterminator="\n")
        self.last_k = -1
        self.ALPHA = 0.90
        self.LENGTH = 4
        self.TIMEOUT = 12
        self.TIMEOUTREAL = self.TIMEOUT/8
        self.prediction_dictionary = {0: "inhale", 1: "exhale", 2: "unknown"}
        self.counter = 0
        self.time = 0
        self.peak = False
        self.last_time = 0


    def process_data(self, data):
        self.peak = False
        self.time +=1
        k = np.argmax(data)
        #print(k)
        z = data[k]
        if k != 2 and z > self.ALPHA:
            if k == self.last_k:
                self.counter += 1
                if self.counter >= self.LENGTH:
                    if self.time -self.last_time > self.TIMEOUT:
                        print(self.prediction_dictionary[k])
                        carrier = [self.prediction_dictionary[k], self.time/8 ]
                        self.peak = True
                        self.writer_object.writerow(carrier)
                        self.counter = 0
                        self.last_time = self.time
            else:
                self.counter = 0

        self.last_k = k
        return self.peak



