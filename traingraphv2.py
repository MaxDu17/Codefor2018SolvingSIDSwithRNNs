import tensorflow as tf
import numpy as np
import random
import os
from make_sets import Setmaker as SM
import csv
class Hyperparameters:
    INPUT_LAYER = 43
    HIDDEN_LAYER = 100 #Modify??
    OUTPUT_LAYER = 3
    NUM_EPOCHS = 10001
    #NUM_EPOCHS = 1
    BATCH_NUMBER = 340
    LEARNING_RATE = 0.1
    VALIDATION_NUMBER = 30
    TEST_NUMBER = 30

class Information:
    INPUT_DIMENSIONS = 43
    INPUT_TIME_DIV = 0.125
    INPUT_SECTORS = 8
    SAMPLE_RATE = 4096

set_maker = SM()
HYP = Hyperparameters()
prediction_dictionary = {0:"inhale", 1:"exhale", 2:"unknown"}

W_In = tf.Variable(tf.random_normal(shape = [HYP.INPUT_LAYER,HYP.HIDDEN_LAYER], stddev = 0.1, mean = 0 ),name = "W_In")#note: this used to have a mean of zero, so check that
W_Hidd =tf.Variable(tf.random_normal(shape = [HYP.HIDDEN_LAYER,HYP.HIDDEN_LAYER], stddev = 0.1, mean =0  ),name = "W_Hidd")
W_Out =tf.Variable(tf.random_normal(shape = [HYP.HIDDEN_LAYER,HYP.OUTPUT_LAYER], stddev = 0.1, mean = 0),name = "W_Out")

B_In = tf.Variable(tf.zeros(HYP.HIDDEN_LAYER),name = "B_In")
B_Hidd = tf.Variable(tf.zeros(HYP.HIDDEN_LAYER), name = "B_Hidd")
B_Out = tf.Variable(tf.zeros(HYP.OUTPUT_LAYER), name = "B_Out")

with tf.name_scope("placeholders"):
    X = tf.placeholder(shape=[1,HYP.INPUT_LAYER],name = "input_placeholder",dtype = tf.float32)
    Y = tf.placeholder(shape=[1,HYP.OUTPUT_LAYER],name = "one_hot_labels_placeholder",dtype = tf.int8)
    last_hidd = tf.placeholder(shape=[1,HYP.HIDDEN_LAYER],name = "previous_hidden_layer_placeholder", dtype = tf.float32)

#with tf.device("/device:GPU:0"):
with tf.name_scope("input_propagation"):
    hidd_layer = tf.matmul(X,W_In)
    hidd_layer = tf.add(hidd_layer,B_In)

with tf.name_scope("hidden_propagation"):
    propagated_prev_hidd_layer = tf.matmul(last_hidd,W_Hidd)
    propagated_prev_hidd_layer = tf.add(propagated_prev_hidd_layer,B_Hidd)
    concat_hidd_layer = tf.add(hidd_layer,propagated_prev_hidd_layer)
    concat_hidd_layer = tf.sigmoid(concat_hidd_layer)
    next_hidd_layer = concat_hidd_layer

with tf.name_scope("logit_output"):
    output_logit = tf.matmul(concat_hidd_layer,W_Out)
    output_logit = tf.add(output_logit, B_Out)

with tf.name_scope("prediction_and_loss"):
    output_prediction = tf.nn.softmax(output_logit, name="output")
    loss = tf.nn.softmax_cross_entropy_with_logits(logits = output_logit,labels=Y,name = "sparse_softmax_loss_function")
    total_loss = tf.reduce_mean(loss)
with tf.name_scope("train"):
    optimizer = tf.train.AdagradOptimizer(learning_rate=HYP.LEARNING_RATE).minimize(total_loss)

with tf.name_scope("summaries_and_saver"):

    tf.summary.histogram("W_Hidd", W_Hidd)
    tf.summary.histogram("W_In", W_In)
    tf.summary.histogram("W_Out", W_Out)

    tf.summary.histogram("B_Hidd", B_Hidd)
    tf.summary.histogram("B_In", B_In)
    tf.summary.histogram("B_Out", B_Out)

    tf.summary.scalar("Loss_at_sample",total_loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

with tf.Session() as sess:
    log_loss = open("Graphv2/GRAPHS/LOSS.csv", "w")
    logger = csv.writer(log_loss, lineterminator="\n")
    '''ckpt = tf.train.get_checkpoint_state(os.path.dirname('GRAPHCHECKPOINTS/'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)'''
    sess.run(tf.global_variables_initializer())
    tf.train.write_graph(sess.graph_def, '.', 'Graphv2/GRAPHS/graph.pbtxt')
    writer = tf.summary.FileWriter("Graphv2/GRAPHS/",sess.graph)
    set_maker.get_test_set()
    total_loss_ = 0

    label = ""
    for epoch in range(HYP.NUM_EPOCHS):
        set_maker.load_next_epoch()
        summed_loss = 0
        for batch_number in range(HYP.BATCH_NUMBER):

            input_array,label = set_maker.load_next_train_sample(batch_number = batch_number)
            one_hot_label = set_maker.one_hot_from_label(label=label)


            one_hot_label = np.reshape(one_hot_label,[1,3])

            counter = 0
            first = True
            for slice in input_array:

                slice = np.reshape(slice,[1,43])

                if counter == 15:

                    next_hidd_layer_,output_logit_,output_prediction_,loss_,total_loss_,summary,_ = sess.run([next_hidd_layer,
                                                                                    output_logit,
                                                                                    output_prediction,
                                                                                    loss,
                                                                                    total_loss,
                                                                                    summary_op,
                                                                                    optimizer], feed_dict=
                    {
                        X: slice,
                        Y: one_hot_label,
                        last_hidd: prev_hidd_layer_
                    })
                    summed_loss += total_loss_
                else:
                    if(first):
                        prev_hidd_layer_ = np.zeros(shape = HYP.HIDDEN_LAYER)
                        prev_hidd_layer_ = np.reshape(prev_hidd_layer_,[1,100])
                        first = False

                    next_hidd_layer_ = sess.run(next_hidd_layer,feed_dict =
                    {
                        X: slice,
                        last_hidd: prev_hidd_layer_
                    })
                    prev_hidd_layer_ = next_hidd_layer_
                    counter+= 1
        logger.writerow(summed_loss)
        print("I have finished epoch ",epoch, " out of ", HYP.NUM_EPOCHS)
        print("the total loss of the last sample in this batch is ", total_loss_)
        print("here is the large sum of losses through the entire epoch: ", summed_loss)

        writer.add_summary(summary,global_step=epoch)

        if epoch % 10 == 0:
            prediction_index = np.argmax(output_prediction_)
            print("Blind Sample Here:")
            print("here is the softmaxed result: ", output_prediction_)
            result = prediction_dictionary[prediction_index]
            print("predicted class: ", result)
            print("real class: ", label)

        if epoch%500 ==0:
            saver.save(sess, "Graphv2/CHECKPOINTS/GraphV2",global_step = epoch)

        if epoch%50 == 0:
            one_hot_label = []
            confusion_matrix = np.zeros([3,3])

            for validation in range(HYP.VALIDATION_NUMBER):
                input_array, label = set_maker.next_validation(batch_number=validation)
                one_hot_label = set_maker.one_hot_from_label(label=label)
                one_hot_label = np.reshape(one_hot_label, [1, 3])
                counter = 0
                first = True
                for slice in input_array:
                    slice = np.reshape(slice, [1, 43])

                    if counter == 15:

                        next_hidd_layer_, output_logit_, output_prediction_, loss_, total_loss_, summary, _ = sess.run(
                            [next_hidd_layer,
                             output_logit,
                             output_prediction,
                             loss,
                             total_loss,
                             summary_op,
                             optimizer], feed_dict=
                            {
                                X: slice,
                                Y: one_hot_label,
                                last_hidd: prev_hidd_layer_
                            })
                    else:
                        if (first):
                            prev_hidd_layer_ = np.zeros(shape=HYP.HIDDEN_LAYER)
                            prev_hidd_layer_ = np.reshape(prev_hidd_layer_, [1, 100])
                            first = False

                        next_hidd_layer_ = sess.run(next_hidd_layer, feed_dict=
                        {
                            X: slice,
                            last_hidd: prev_hidd_layer_
                        })
                        prev_hidd_layer_ = next_hidd_layer_
                        counter += 1
                prediction_index = np.argmax(output_prediction_)
                label_index = np.argmax(one_hot_label)
                confusion_matrix[prediction_index][label_index] +=1
            print(confusion_matrix)


    one_hot_label = []
    confusion_matrix = np.zeros([3, 3])
    for test in range(HYP.TEST_NUMBER):
        input_array, label = set_maker.next_validation(batch_number=test)
        one_hot_label = set_maker.one_hot_from_label(label=label)
        one_hot_label = np.reshape(one_hot_label, [1, 3])
        counter = 0
        first = True
        for slice in input_array:
            slice = np.reshape(slice, [1, 43])

            if counter == 15:

                next_hidd_layer_, output_logit_, output_prediction_, loss_, total_loss_, summary, _ = sess.run(
                    [next_hidd_layer,
                     output_logit,
                     output_prediction,
                     loss,
                     total_loss,
                     summary_op,
                     optimizer], feed_dict=
                    {
                        X: slice,
                        Y: one_hot_label,
                        last_hidd: prev_hidd_layer_
                    })
            else:
                if (first):
                    prev_hidd_layer_ = np.zeros(shape=HYP.HIDDEN_LAYER)
                    prev_hidd_layer_ = np.reshape(prev_hidd_layer_, [1, 100])
                    first = False

                next_hidd_layer_ = sess.run(next_hidd_layer, feed_dict=
                {
                    X: slice,
                    last_hidd: prev_hidd_layer_
                })
                prev_hidd_layer_ = next_hidd_layer_
                counter += 1
        prediction_index = np.argmax(output_prediction_)
        label_index = np.argmax(one_hot_label)
        confusion_matrix[prediction_index][label_index] += 1
    print(confusion_matrix)
    writer.close()
