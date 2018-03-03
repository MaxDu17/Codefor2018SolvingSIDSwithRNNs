import tensorflow as tf
import os
import numpy as np

pbfilename = "Best/modelv1.pb"

with tf.gfile.GFile(pbfilename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
