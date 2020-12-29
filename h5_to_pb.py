import logging
import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from tensorflow.python.keras import backend as K
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

# necessary !!!
tf.compat.v1.disable_eager_execution()

h5_path = '/media/thanglmb/Bkav/AICAM/TrainModels/TF2/Classifier/ResNet50.h5'
model = keras.models.load_model(h5_path)
model.summary()

'''
# save pb
with K.get_session() as sess:
    output_names = [out.op.name for out in model.outputs]
    input_graph_def = sess.graph.as_graph_def()
    for node in input_graph_def.node:
        node.device = ""
    graph = graph_util.remove_training_nodes(input_graph_def)
    graph_frozen = graph_util.convert_variables_to_constants(sess, graph, output_names)
    tf.io.write_graph(graph_frozen, '/media/thanglmb/DATA/MyProject/AI_DL/DogCat-Classifier/TF/frozen_graph/model','train.pb', as_text=False)
logging.info("save pb successfully！")
'''

with K.get_session() as sess:
    output_names = [out.op.name for out in model.outputs]
    input_graph_def = sess.graph.as_graph_def()
    for node in input_graph_def.node:
        node.device = ""
    graph = tf.compat.v1.graph_util.remove_training_nodes(input_graph_def)
    graph_frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph, output_names)
    tf.io.write_graph(graph_frozen, '/media/thanglmb/Bkav/AICAM/TrainModels/TF2/Classifier','ResNet50.pb', as_text=False)
logging.info("save pb successfully！")
