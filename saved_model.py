import tensorflow as tf

pb_saved_model = "exported/"
OUTPUT_NAMES = ["final_result"]

_graph = tf.Graph()
with _graph.as_default():
    _sess = tf.Session(graph=_graph)
    model = tf.saved_model.loader.load(_sess, ["serve"], pb_saved_model)
    graphdef = tf.get_default_graph().as_graph_def()
    frozen_graph = tf.graph_util.convert_variables_to_constants(_sess, graphdef, OUTPUT_NAMES)
    #frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

with tf.gfile.GFile("exported/frozen.pb", "wb") as f:
    f.write(frozen_graph.SerializeToString())
