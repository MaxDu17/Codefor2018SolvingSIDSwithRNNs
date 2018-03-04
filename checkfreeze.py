import tensorflow as tf
g = tf.GraphDef()
g.ParseFromString(open("Best/modelv1.pb", "rb").read())
k = [n for n in g.node if n.name.find("input") != -1] # same for output or any other node you want to make sure is ok
p = g.node
print(p)