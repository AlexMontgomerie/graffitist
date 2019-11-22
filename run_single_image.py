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
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" # e.g. GPUs 0 and 1
# LOAD GRAFFITIST KERNELS
kernel_root = os.path.join(os.path.dirname(graffitist.__file__), 'kernels')
if tf.test.is_built_with_cuda() and tf.test.is_gpu_available(cuda_only=True):
  tf.load_op_library(os.path.join(kernel_root, 'quantize_ops_cuda.so'))
else:
  tf.load_op_library(os.path.join(kernel_root, 'quantize_ops.so'))

parser = argparse.ArgumentParser(description='Single Image Script')
# DATASET PATH
parser.add_argument('--image_path', metavar='PATH', required=True,
          help='path to image')
# MODEL NAME
parser.add_argument('--model_name', required=True,
          help='model name, to be used for outputs')
# GRAPH PATH
parser.add_argument('--graph_path', metavar='PATH', required=True,
          help='path to the frozen model.pb')
# CKPT PATH
parser.add_argument('--ckpt_path', metavar='PATH', required=True,
          help='path to the .ckpt files')
# INPUT NODE
parser.add_argument('-i', '--input_node', metavar='STRING', type=str, required=True,
          help='input node')
# OUTPUT NODE
parser.add_argument('-o', '--output_node', metavar='STRING', type=str, required=True,
          help='output node')
args = parser.parse_args()

# DEFINE MODEL VARIABLES
MODEL_NAME    = args.model_name
GRAPH_PB_PATH = args.graph_path
CKPT_PATH     = args.ckpt_path

#  DEFINE DATASET VARIABLES
IMAGE_PATH = args.image_path

# NODES IN AND OUT
INPUT_NODE  = args.input_node
OUTPUT_NODE = args.output_node
 
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

  # define input and output nodes
  l_input  = tf.get_default_graph().get_tensor_by_name(INPUT_NODE+":0")
  l_output = tf.get_default_graph().get_tensor_by_name(OUTPUT_NODE+":0")

  # load input image
  image, label = imagenet_utils.get_image(IMAGE_PATH, CKPT_PATH, 224) # NOTE: might need to change image size
  image = tf.expand_dims(image,0).eval()
  # run net and get output probabilities
  output_dict = sess.run(l_output,feed_dict = {l_input : image})
 
