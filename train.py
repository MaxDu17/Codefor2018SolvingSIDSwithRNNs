import tensorflow as tf
import numpy as np
import random
import os

class Hyperparameters:
    INPUT_LAYER = 1
    HIDDEN_LAYER = 50 #Modify??
    OUTPUT_LAYER = 1
    NUM_EPOCHS = 5000
    LEARNING_RATE = 0.1
    
class Information:
    INPUT_DIMENSIONS = 43
    INPUT_TIME_DIV = 0.125
    INPUT_SECTORS = 8
    SAMPLE_RATE = 4096
