from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.platform import gfile
import graffitist
import numpy as np
import os
import scripts.imagenet_utils as imagenet_utils
import json
import csv
from tqdm import tqdm
import uuid
import argparse
import math

# LIMIT GPUs
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# LOAD GRAFFITIST KERNELS
kernel_root = os.path.join(os.path.dirname(graffitist.__file__), 'kernels')
if tf.test.is_built_with_cuda() and tf.test.is_gpu_available(cuda_only=True):
  tf.load_op_library(os.path.join(kernel_root, 'quantize_ops_cuda.so'))
else:
  tf.load_op_library(os.path.join(kernel_root, 'quantize_ops.so'))

parser = argparse.ArgumentParser(description='Validation Script')
# DATASET PATH
parser.add_argument('--data_dir', metavar='PATH', required=True,
          help='path to imagenet validation dataset dir')
# MODEL NAME
parser.add_argument('--model_name', required=True,
          help='model name, to be used for outputs')
# GRAPH PATH
parser.add_argument('--graph_path', metavar='PATH', required=True,
          help='path to the frozen model.pb')
# CKPT PATH
parser.add_argument('--ckpt_path', metavar='PATH', required=True,
          help='path to the .ckpt files')
# BATCH SIZE
parser.add_argument('-b', '--batch_size', type=int, default=100, metavar='N',
          help='mini-batch size (default: 100)')

args = parser.parse_args()

# DEFINE MODEL VARIABLES
MODEL_NAME    = args.model_name
GRAPH_PB_PATH = args.graph_path
CKPT_PATH     = args.ckpt_path

#  DEFINE DATASET VARIABLES
DATASET_PATH = args.data_dir 
BATCH_SIZE   = args.batch_size
 
# open synset-class look up table
with open('synset_idx.json','r') as f:
  synset_idx = json.load(f)


# START TENSORFLOW SESSION 
with tf.compat.v1.Session() as sess:

  # INIT TF VARIABLES 
  sess.run(tf.global_variables_initializer())
 
  # LOAD GRAPH
  with tf.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
  graph_def.ParseFromString(f.read())
  sess.graph.as_default()
  tf.import_graph_def(graph_def, name='')

  # LOAD WEIGHTS
  var_list = {}
  reader = tf.compat.v1.train.NewCheckpointReader(CKPT_PATH)
  for key in reader.get_variable_to_shape_map():
    # Look for all variables in ckpt that are used by the graph
    try:
      tensor = sess.graph.get_tensor_by_name(key + ":0")
    except KeyError:
      # This tensor doesn't exist in the graph (for example it's
      # 'global_step' or a similar housekeeping element) so skip it.
      continue
    var_list[key] = tensor
  saver = tf.compat.v1.train.Saver(var_list=var_list)
  saver.restore(sess, CKPT_PATH)

  # Get the dataset
  features, labels, filenames, _ = imagenet_utils.dataset_input_image_fn(DATASET_PATH, GRAPH_PB_PATH, 224, BATCH_SIZE, 8,filenames=True)

  # define input and output nodes
  l_input  = tf.get_default_graph().get_tensor_by_name("input:0")
  l_output = tf.get_default_graph().get_tensor_by_name("MobilenetV1/Predictions/Softmax:0")

  # Reset output directory
  if os.path.isdir('outputs/'+MODEL_NAME):
    for filename in os.listdir('outputs/'+MODEL_NAME):
      os.remove('outputs/'+MODEL_NAME+'/'+filename)
  else:
    os.mkdir('outputs/'+MODEL_NAME)

  # Run
  for i in tqdm(range(math.ceil(50000/BATCH_SIZE))):
    # Get image batch & true label
    image, label, image_filename = sess.run([features, labels, filenames])
    assert image.shape == (BATCH_SIZE,224,224,3), "Image wrong shape"
    assert label.shape == (BATCH_SIZE,), "Label wrong shape"
    # get output probabilities
    output_dict = sess.run(l_output,feed_dict = {l_input : image})
    # create filename
    filename = 'outputs/'+MODEL_NAME+'/'+uuid.uuid4().hex+'.csv'
    # Write to output csv
    #   [ true_label, prob[0], prob[1], ... ]
    with open(filename,'w') as f:
      writer = csv.writer(f,delimiter=',')
      for i in range(len(label)):
        tmp = [ image_filename[i].decode("utf-8"), synset_idx[label[i].decode("utf-8")] ]
        tmp.extend(output_dict[i].tolist())
        writer.writerow( tmp )

