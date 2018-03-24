import scipy as sp
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import sys
class Trendtest:

    def __init__(self):
        self.big_list = list()
        self.time_counter = list()
        self.lower_bound = 5
        self.counter = 0
        self.alpha = 0.15

        plt.figure(num="graph")
        plt.ion()

        axes = plt.gca()
        axes.set_xlim(0, 5)
        axes.set_ylim(0, 10)

    def significance(self, number):
        self.big_list.append(number)
        self.counter += 1
        self.time_counter.append(self.counter)

        if len(self.big_list) > self.lower_bound:
            std = np.std(self.big_list)
            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(self.time_counter, self.big_list)
            carrier_number = (self.counter-1)**0.5
            std_of_slope = std_err/(std*carrier_number)
            area = self.alpha/2
            critical_value = t.ppf(area, self.counter-1)
            critical_value = critical_value *-1
            plus_minus_val = critical_value*std_of_slope
            big = slope + plus_minus_val
            small = slope - plus_minus_val
            print("upper bound: ", big)
            print("lower bound: ", small)
            self.plot(big, small, slope)
            if 0 > big or 0 < small:
                print("ALARM ALARM ALARM ALARM")

    def test_library(self):
        list = [10,12,10,14,13,12,15,16,17,18,19,20]
        alternate_list = [10,11,10,9,10,10,10,10,11,10,9,9,0]
        for i in range(12):
            self.significance(alternate_list[i])
    def plot(self, upper_bound, lower_bound, slope):
        axes = plt.gca()
        axes.set_xlim(0, 1)
        axes.set_ylim(-0.2, 0.2)
        plt.bar(0.1, upper_bound, width = 0.2, color = "0.5")
        plt.bar(0.3, lower_bound,width = 0.2, color="0.1")
        plt.bar(0.7, slope, width=0.2, color=(0,0,1))
        plt.pause(1)
        plt.cla()

    def flush(self):
        del self.big_list[:]
        del self.time_counter[:]
        self.counter = 0

test = Trendtest()
test.test_library()