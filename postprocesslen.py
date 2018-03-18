import numpy as np
import csv

class Processor:

    def __init__(self, file_name):
        p = open(file_name, "w")
        self.writer_object = csv.writer(p,lineterminator="\n")
        self.last_k = -1
        self.ALPHA = 0.98
        self.LENGTH = 5
        self.prediction_dictionary = {0: "inhale", 1: "exhale", 2: "unknown"}
        self.counter = 0
        self.time = 0


    def process_data(self, data):
        self.time +=1
        k = np.argmax(data)
        z = data[k]
        if k != 2 and z > self.ALPHA:
            if k == self.last_k:
                self.counter += 1
                if self.counter >= self.LENGTH:
                    print(self.prediction_dictionary[k])
                    carrier = [self.prediction_dictionary[k], self.time/8 ]
                    self.writer_object.writerow(carrier)
                    self.counter = 0
            else:
                self.counter = 0

        self.last_k = k



