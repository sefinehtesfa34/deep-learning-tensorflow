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
