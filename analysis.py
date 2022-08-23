import numpy as np
from keras.utils.np_utils import to_categorical
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import random
#import av
import os
import sys
import numpy as np
#import skimage as ski
import skimage.feature
import matplotlib.pyplot as plt
from PIL import Image
import random

import tensorflow as tf
from tensorflow import keras
import mat73
matFilename = "C:/Users/11924/Documents/Matlab/STFT/stft_input_2.mat"
mat = mat73.loadmat(matFilename)
dataset = mat['result']
y=mat['y']
del mat
dataset.astype(np.float16)
print(dataset.shape)
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
scaler = MinMaxScaler(feature_range=(0, 1))
k_i = 0
recall_score_i = []
precision_score_i = []
f1_score_i = []
recall_score_i_train = []
precision_score_i_train = []
f1_score_i_train = []
for train_index, test_index in kfold.split(y):
  if k_i in [0,1,2]:
    k_i+=1
    continue
  model = tf.keras.models.load_model('content/output_5/Fourier_2D_model_'+str(k_i)+'.h5')
  X_train = dataset[train_index,:,:]
  y_train = y[train_index]
  X_test = dataset[test_index, :, :]
  y_test = y[test_index]
  y_train_pred = model.predict(X_train)
  print(k_i)
  y_test_pred = model.predict(X_test)
  print(k_i)
  train_label = np.argmax(y_train_pred,axis=1)
  test_label = np.argmax(y_test_pred, axis=1)
  # recall_score_i.append(sklearn.metrics.recall_score(y_test,test_label,average='macro'))
  # precision_score_i.append(sklearn.metrics.precision_score(y_test,test_label,average='macro'))
  # f1_score_i.append(sklearn.metrics.f1_score(y_test,test_label,average='macro'))
  #
  # recall_score_i_train.append(sklearn.metrics.recall_score(y_train,train_label,average='macro'))
  # precision_score_i_train.append(sklearn.metrics.precision_score(y_train,train_label,average='macro'))
  # f1_score_i_train.append(sklearn.metrics.f1_score(y_train,train_label,average='macro'))
  k_i+=1

  np.save('output_pred/train_' + str(k_i) + '.npy', train_label)
  np.save('output_pred/test_' + str(k_i) + '.npy', test_label)

# print(np.mean(recall_score_i))
# print(np.std(recall_score_i))
# print(np.mean(precision_score_i))
# print(np.std(precision_score_i))
# print(np.mean(f1_score_i))
# print(np.std(f1_score_i))
#
# print(np.mean(recall_score_i_train))
# print(np.std(recall_score_i_train))
# print(np.mean(precision_score_i_train))
# print(np.std(precision_score_i_train))
# print(np.mean(f1_score_i_train))
# print(np.std(f1_score_i_train))