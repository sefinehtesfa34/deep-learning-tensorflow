import numpy as np

import typing
from typing import Any, Tuple

import tensorflow as tf

import tensorflow_text as tf_text

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pathlib

# Download the file
url="http://www.manythings.org/anki/spa-eng.zip"
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin=url,
    extract=True)
path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'
def load_data(path):
  text = path.read_text(encoding='utf-8')

  lines = text.splitlines()
  pairs = [line.split('\t') for line in lines]

  inp = [inp for targ, inp in pairs]
  targ = [targ for targ, inp in pairs]

  return targ, inp

# Create a tf.data dataset
# From these arrays of strings 
# you can create a tf.data.Dataset of strings 
# that shuffles and batches them efficiently:

targ, inp = load_data(path_to_file)
BUFFER_SIZE = len(inp)
BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
