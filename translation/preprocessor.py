import numpy as np
import os 
import typing
from typing import Any, Tuple

import tensorflow as tf

import tensorflow_text as tf_text

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pathlib


# Download the file
path="spa-eng/spa.txt"
fullPath = os.path.abspath("./" + path)
text_file_path = tf.keras.utils.get_file("spa.txt", 'file://'+fullPath)

path_to_file = pathlib.Path(text_file_path)
def load_data(path):

  text = path.read_text(encoding='utf-8')

  lines = text.splitlines()
  pairs = [line.split('\t')[:2] for line in lines]
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

# Standardization
# The model is dealing with multilingual text with a limited vocabulary. 
# So it will be important to standardize the input text.
# The first step is Unicode normalization to split accented characters 
# and replace compatibility characters with their ASCII equivalents.
# The tensorflow_text package contains a unicode normalize operation:

# Unicode normalization will be the first step in the text standardization function:
def tf_lower_and_split_punct(text):
  # Split accecented characters.
  text = tf_text.normalize_utf8(text, 'NFKD')
  text = tf.strings.lower(text)
  # Keep space, a to z, and select punctuation.
  text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
  # Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
  # Strip whitespace.
  text = tf.strings.strip(text)

  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text
