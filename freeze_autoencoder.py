import tensorflow as tf
from tensorflow.python.tools import freeze_graph
MODEL_NAME = 'Audoencoder_frozen'

# Freeze the graph

input_graph_path = "GraphV6/autoencoder.pbtxt"
checkpoint_path = "GraphV6/CHECKPOINTS_AUTO/GraphV6-2000"
input_saver_def_path = ""
input_binary = False
output_node_names = "encoding/compression"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'GraphV6/'+MODEL_NAME+'.pb'

clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")