import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt
import seaborn as sns

layer = ''
for i in range(1, 4):
  feature_path = "content/CNN_feature_"+str(i)+'.npy'
  tmp = np.load(feature_path)
  if layer == '':
    layer = tmp
  else:
    layer = np.vstack((layer, tmp))

del tmp

y_test = np.load("content/CNN_y_test.npy")
list = np.load("content/CNN_list.npy")

print(y_test.shape)
print(list.shape)
print(layer.shape)

# dimensionality reduction

# buf=np.array(layer)
buf = layer
tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(buf)

y_test=y_test.reshape(y_test.shape[0],1)
# list=np.array(list)
list=list.reshape(list.shape[0],1)
tsne_result=np.hstack((tsne_result, y_test))
np.save("content/CNN_tsne", tsne_result)


# check the predictions
for i in range(0,tsne_result.shape[0]):
  if tsne_result[i][2]==list[i]:
    list[i]=0
  else:
    list[i]=1
# map (0,1) to (Correct,Error) and record in list correct[]
correct=[]
size=[]  # point size in graph
for i in range(0, len(list)):
  if list[i]==0:
    correct.append("Correct")
  else:
    correct.append("Error")
  size.append(1)

df = pd.DataFrame(tsne_result, columns = ['x','y', 'Stages'])
df['Correctness'] = correct
df['size'] = size
df.to_csv(r'content/CNN_df.csv', index=False)
print(df.head(15))