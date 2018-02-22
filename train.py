import tensorflow as tf
import numpy as np
import random
import os

class Hyperparameters:
    INPUT_LAYER = 43
    HIDDEN_LAYER = 100 #Modify??
    OUTPUT_LAYER = 3
    NUM_EPOCHS = 5000
    LEARNING_RATE = 0.1

class Information:
    INPUT_DIMENSIONS = 43
    INPUT_TIME_DIV = 0.125
    INPUT_SECTORS = 8
    SAMPLE_RATE = 4096

    


HYP = Hyperparameters()

W_In = tf.Variable(tf.random_normal(shape = [HYP.INPUT_LAYER,HYP.HIDDEN_LAYER], stddev = 0.1 ))#note: this used to have a mean of zero, so check that
W_Hidd =tf.Variable(tf.random_normal(shape = [HYP.HIDDEN_LAYER,HYP.HIDDEN_LAYER], stddev = 0.1 ))
W_Out =tf.Variable(tf.random_normal(shape = [HYP.HIDDEN_LAYER,HYP.OUTPUT_LAYER], stddev = 0.1 ))

B_In = tf.Variable(tf.zeros(HYP.HIDDEN_LAYER))
B_Hidd = tf.Variable(tf.zeros(HYP.HIDDEN_LAYER))
B_Out = tf.Variable(tf.zeros(HYP.OUTPUT_LAYER))