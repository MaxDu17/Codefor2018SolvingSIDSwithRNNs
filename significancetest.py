import scipy as sp
import numpy as np
from scipy.stats import t
import sys
class Trendtest:

    def __init__(self):
        self.big_list = list()
        self.time_counter = list()
        self.lower_bound = 5
        self.counter = 0
        self.alpha = 0.15

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
            if 0 > big or 0 < small:
                print("ALARM ALARM ALARM ALARM")

    def test_library(self):
        list = [10,12,10,14,13,12,15,16,17,18,19,20]
        alternate_list = [10,11,10,9,10,10,10,10,11,10,9,9,0]
        for i in range(12):
            self.significance(alternate_list[i])


    def flush(self):
        del self.big_list[:]
        del self.time_counter[:]
        self.counter = 0

test = Trendtest()
test.test_library()