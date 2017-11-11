import tensorflow as tf, sys

image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# To load the label of the test image from the label file
# retrained_list.txt will contain all the labels (list of classes in the TinyImagenet challenge)
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

#T o grab our model from the saved and retrained graph file
# Unpersists graph from file
with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# We now have the image and model ready.
# Time for prediction using our retrained classifier

# Creation of a Tensorflow session
# TF Session offers us an environment to perform operations on the data
with tf.Session() as sess:

    # Softmax_Tensor function works on the last layer of the model
    # It uses the final layer to map the input data into probabilities of an expected output
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # Feed the image_data as input to the graph and get first prediction
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    big = None
    spot = None

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]

        # Rather than listing the confidence score for all the classes, print only the class with the maximum confidence score
        if big is None or score > big:
            big = score
            spot = label_lines[node_id]
            print('%s, %s, (score = %.5f)' % (image_path, spot, big))
