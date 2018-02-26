import numpy as np
confusion_matrix =np.zeros((3, 3))
print(confusion_matrix)
confusion_matrix[0][1] += 1
print(confusion_matrix)