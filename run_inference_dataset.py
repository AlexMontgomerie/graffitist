from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os
from tensorflow.python.platform import gfile
import graffitist
from PIL import Image
import scripts.imagenet_utils as imagenet_utils
import re
import json
from multiprocessing import Pool
import multiprocessing
from joblib import Parallel, delayed
import csv
from tqdm import tqdm
import uuid


# LIMIT GPUs
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# LOAD GRAFFITIST KERNELS
kernel_root = os.path.join(os.path.dirname(graffitist.__file__), 'kernels')
if tf.test.is_built_with_cuda() and tf.test.is_gpu_available(cuda_only=True):
  tf.load_op_library(os.path.join(kernel_root, 'quantize_ops_cuda.so'))
else:
  tf.load_op_library(os.path.join(kernel_root, 'quantize_ops.so'))

# DEFINE MODEL PATHS
MODEL_NAME    = 'vgg16_slim_pretrained'
GRAPH_PB_PATH = './models/vgg16_slim_pretrained/vgg16_slim_pretrained_quant.pb'
CKPT_PATH     = './models/vgg16_slim_pretrained/vgg16_slim_pretrained_graffitist.ckpt'
GRAPH_PB_PATH = './models/vgg16_slim_pretrained/vgg16_slim_pretrained.pb'
CKPT_PATH     = './models/vgg16_slim_pretrained/vgg16_slim_pretrained.ckpt'

# DEFINE DATASET PATH
DATASET_PATH = '/data/imagenet/validation/'
BATCH_SIZE = 100
# top 1 and 5
top1 = 0
top5 = 0
total_images = 0
 
# open synset-class look up table
with open('synset_idx.json','r') as f:
  synset_idx = json.load(f)

# START TENSORFLOW SESSION 
with tf.compat.v1.Session() as sess:
  
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

  # helper function to process images
  def _process_image(image):
    label = tf.strings.split(image,'/',result_type="RaggedTensor")[-2]
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = imagenet_utils.vgg_preprocess_input_fn(image, 224, 224) # TODO: must change function for different networks
    return image, label

  # preprocess images
  dataset = tf.data.Dataset.list_files(DATASET_PATH+'*/*')
  dataset = dataset.map(_process_image, num_parallel_calls=4)
  dataset = dataset.batch(BATCH_SIZE)
  dataset = dataset.prefetch(buffer_size=1)

  # get iterator
  dataset = dataset.make_one_shot_iterator()
  features, labels = dataset.get_next()

  # define input and output nodes
  l_input  = tf.get_default_graph().get_tensor_by_name("input:0")
  l_output = tf.get_default_graph().get_tensor_by_name("vgg_16/fc8/squeezed:0")

  # TODO: create dir and remove all existing files 
  if os.path.isdir('outputs/'+MODEL_NAME):
    for filename in os.listdir('outputs/'+MODEL_NAME):
      os.remove('outputs/'+MODEL_NAME+'/'+filename)
  else:
    os.mkdir('outputs/'+MODEL_NAME)

  # Run
  for i in tqdm(range(int(50000/BATCH_SIZE))):
    # Get image batch & true label
    image, label = sess.run([features, labels])
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
        tmp = [ synset_idx[label[i].decode("utf-8")] ]
        tmp.extend(output_dict[i].tolist())
        writer.writerow( tmp )
'''
  # run the network
  values, labels = tf.nn.top_k(output_dict,k=5)
  

  total_images += 50

  for label in labels.eval():
    if idx == label[0]:
      top1 += 1
    if idx in label:
      top5 += 1
    
  print("top1: ",top1/total_images, ", top5: ",top5/total_images)
  # save info 


print("top1: ",top1/50000, ", top5: ",top5/50000)

print('done!')
'''
