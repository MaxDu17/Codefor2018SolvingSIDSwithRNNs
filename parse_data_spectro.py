from parse_data import DataParse as pd
import matplotlib.pyplot as plt
import numpy as np
from make_sets import Source as SS

source_dir = SS()
dataparser = pd()

name = source_dir.Native.INHALE_DIR + "52" +".wav"
matrix = dataparser.prepare_data(name)
numbers = np.linspace(0,15,16)
new_matrix =np.transpose(matrix)

plt.figure(num='INHALE')
for i in range(43):
    plt.plot(numbers, new_matrix[i], label = str(i))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=10, mode="expand", borderaxespad=0.)

name = source_dir.Native.EXHALE_DIR + "46" +".wav"
matrix = dataparser.prepare_data(name)
numbers = np.linspace(0,15,16)
new_matrix =np.transpose(matrix)

plt.figure(num='EXHALE')
for i in range(43):
    plt.plot(numbers, new_matrix[i], label = str(i))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=10, mode="expand", borderaxespad=0.)

plt.show()