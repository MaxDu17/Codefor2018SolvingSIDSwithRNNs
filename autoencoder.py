from parse_data import DataParse as DP
import tensorflow as tf
import numpy as np
import random
Parser = DP()

class Hyperparameters:
    input = 512
    first = 128
    compression = 50
    third = 128
    output = 512

    BATCH_NUMBER = 300
    LEARNING_RATE = 0.08
    EPOCHS = 2001
    TOTAL = 800

class Source:
    class Current:
        INHALE_DIR = "sen_data/inhale/"
        EXHALE_DIR = "sen_data/exhale/"
        UNKNOWN_DIR= "sen_data/unknown/"
    class Native:
        INHALE_DIR = "sen_data/inhale/"
        EXHALE_DIR = "sen_data/exhale/"
        UNKNOWN_DIR= "sen_data/unknown/"

HYP = Hyperparameters()
W_In = tf.Variable(tf.random_normal(shape = [HYP.input,HYP.first], stddev = 0.1, mean = 0,name = "W_In"))
W_First = tf.Variable(tf.random_normal(shape = [HYP.first,HYP.compression], stddev = 0.1, mean = 0,name = "W_First"))
W_Compression = tf.Variable(tf.random_normal(shape = [HYP.compression,HYP.third], stddev = 0.1, mean = 0,name = "W_Compression"))
W_Third = tf.Variable(tf.random_normal(shape = [HYP.third,HYP.output], stddev = 0.1, mean = 0,name = "W_Third"))

B_In = tf.Variable(tf.zeros(shape = [HYP.first], name = "B_In"))
B_First = tf.Variable(tf.zeros(shape = [HYP.compression],name = "B_First"))
B_Compression = tf.Variable(tf.zeros(shape = [HYP.third],name = "B_Compression"))
B_Third = tf.Variable(tf.zeros(shape = [HYP.output],name = "B_Third"))

with tf.name_scope("placeholders"):
    X = tf.placeholder(shape=[1,HYP.input],name = "input_and_label_placeholder",dtype = tf.float32)

with tf.name_scope("encoding"):
    first_layer = tf.tanh(tf.add(tf.matmul(X,W_In), B_In),name = "first_layer")
    compression_layer =  tf.tanh(tf.add(tf.matmul(first_layer,W_First), B_First),name = "compression")

with tf.name_scope("decoding"):
    third_layer = tf.tanh(tf.add(tf.matmul(compression_layer,W_Compression), B_Compression), name = "third_layer")
    output_layer =  tf.tanh(tf.add(tf.matmul(third_layer,W_Third), B_Third),name = "output")

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(tf.subtract(X, output_layer)))

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdagradOptimizer(learning_rate=HYP.LEARNING_RATE).minimize(loss)


with tf.name_scope("summaries_and_saver"):

    tf.summary.histogram("W_In", W_In)
    tf.summary.histogram("W_First", W_First)
    tf.summary.histogram("W_Compression", W_Compression)
    tf.summary.histogram("W_Third", W_Third)

    tf.summary.histogram("B_In", B_In)
    tf.summary.histogram("B_First", B_First)
    tf.summary.histogram("B_Compression", B_Compression)
    tf.summary.histogram("B_Third", B_Third)

    tf.summary.scalar("Loss_at_sample",loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    tf.train.write_graph(sess.graph_def, '/', 'autoencoder.pbtxt')
    writer = tf.summary.FileWriter("GraphV6/GRAPHS/",sess.graph)
    loss = 0
    big_list = [i for i in range(HYP.TOTAL)]
    for i in range(HYP.EPOCHS):
        training_list = random.sample(big_list, HYP.BATCH_NUMBER)
        
        for batch_index in training_list:
            if batch_index <200:
                file_name = Source.Current.INHALE_DIR + str(batch_index) + ".wav"
            elif batch_index >=200 and batch_index < 400:
                file_name = Source.Current.EXHALE_DIR + str(batch_index-200) + ".wav"
            else:

                file_name = Source.Current.UNKNOWN_DIR + str(batch_index - 400) + ".wav"

            loaded_fouriers = Parser.prepare_data_autoencoder(file_name)
            for sample in loaded_fouriers:
                output_layer_, loss_, _ = sess.run([output_layer,loss,optimizer], feed_dict = {X:sample})

