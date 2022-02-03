# TEXT-CLASSIFICATION
This code uses the IMDB sentiment classification dataset (unprocessed); and 'TextVectorization' for word splitting and indexing.
#(I suggest running it on colab.reseach.google.com since its much easier than to go through the full set up process)


#----SETUP--#
import tensorflow as tf
import numpy as np


#--------PREPARE-THE-DATA--------#
#removing tags because the standardizer doesnt strip HTML... making a custom standardization function.

from tensorflow.keras.layers import TextVectorization
import string
import re


#---------BUILD-A-MODEL--------#
#1D covnet with an 'Embedding' layer.
from tensorflow.keras import layers

#--------TRAIN-THE-MODEL-------#

#-----_-EVALUATE-THE-MODEL-------#

#Test it with `raw_test_ds`, which yields raw strings (END-TO-END MODEL)
