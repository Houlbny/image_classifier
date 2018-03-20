import sys
import tensorflow as tf
import os

filepath = sys.argv[1]

f = os.listdir(filepath)

for i in f:
    image_path = filepath + i

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("./labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("./output.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        human_string = label_lines[top_k[0]]
        score = predictions[0][top_k[0]]
        print('---------%s (score = %.5f)-----------' % (human_string, score))

        oldname=image_path

        prename = oldname.split("/")
        #设置新文件名
        newname=prename[0] + "/" + prename[1] + "/" + human_string + prename[2]
         
        #用os模块中的rename方法对文件改名
        os.rename(oldname,newname)
        print(oldname,'======>',newname)

print("All done!!")

