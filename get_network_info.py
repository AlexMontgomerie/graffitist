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

# TODO:
# - improve dataset execution method

# LIMIT GPUs
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# LOAD GRAFFITIST KERNELS
kernel_root = os.path.join(os.path.dirname(graffitist.__file__), 'kernels')
if tf.test.is_built_with_cuda() and tf.test.is_gpu_available(cuda_only=True):
  tf.load_op_library(os.path.join(kernel_root, 'quantize_ops_cuda.so'))
else:
  tf.load_op_library(os.path.join(kernel_root, 'quantize_ops.so'))


