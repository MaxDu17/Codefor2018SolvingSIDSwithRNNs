import tensorflow as tf
from tensorflow.python.tools import freeze_graph
MODEL_NAME = 'modelv1'

# Freeze the graph

input_graph_path = "scratch/tfdroid.pbtxt"
checkpoint_path = "scratch/tfdroid.ckpt"
input_saver_def_path = ""
input_binary = False
output_node_names = "O"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'scratch/frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")