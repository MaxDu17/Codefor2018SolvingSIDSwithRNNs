from parse_data import DataParse as pd
import matplotlib.pyplot as plt
import numpy as np
from make_sets import Source as SS

source_dir = SS()
dataparser = pd()
BINS = 43

name = "sen_data/inhale/" + "3" +".wav"

matrix = dataparser.prepare_data(name)
numbers = np.linspace(0,15,16)
new_matrix =np.transpose(matrix)
plt.figure(num='INHALE')
for i in range(BINS):
    plt.plot(numbers, new_matrix[i], label = str(i))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=10, mode="expand", borderaxespad=0.)


name = "sen_data/unknown/" + "25" +".wav"
matrix = dataparser.prepare_data(name)
numbers = np.linspace(0,15,16)
new_matrix =np.transpose(matrix)
plt.figure(num='NOISE')
for i in range(BINS):
    plt.plot(numbers, new_matrix[i], label = str(i))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=10, mode="expand", borderaxespad=0.)

name = "sen_data/exhale/" + "5" +".wav"

matrix = dataparser.prepare_data(name)
numbers = np.linspace(0,15,16)
new_matrix =np.transpose(matrix)
plt.figure(num='EXHALE')
for i in range(BINS):
    plt.plot(numbers, new_matrix[i], label = str(i))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=10, mode="expand", borderaxespad=0.)

plt.show()