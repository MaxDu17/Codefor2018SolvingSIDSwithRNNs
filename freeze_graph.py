import tensorflow as tf
from tensorflow.python.tools import freeze_graph
MODEL_NAME = 'GraphV2_frozen'

# Freeze the graph

input_graph_path = "GraphV2/GRAPHS/graph.pbtxt"
checkpoint_path = "GraphV2/CHECKPOINTS/GraphV2-10000"
input_saver_def_path = ""
input_binary = False
output_node_names = "prediction_and_loss/output"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'GraphV2/GRAPHS/'+MODEL_NAME+'.pb'

clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")