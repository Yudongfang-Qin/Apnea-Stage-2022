K = 0

import numpy as np
from keras.utils.np_utils import to_categorical
import pandas as pd
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
import numpy as np
#import skimage as ski
import skimage.feature
import matplotlib.pyplot as plt
from PIL import Image
import random
from keras.preprocessing.image import img_to_array, array_to_img
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import sys
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

path = "C:\\Users\\11924\\Documents\\Matlab\\STFT\\2D_input_full"
#path = "dataste"
files= os.listdir(path)
# dataset = np.zeros([len(files),2049,120])
dataset_length = len(files)
dataset = np.zeros([dataset_length,2049,120])
y = []
X = []
for i in range(0,dataset_length):
    file_name = files[i]
    y.append(int(file_name[0]))
    #X.append(MinMaxScaler(pd.read_csv(path+"/"+file,header=None)));
#index = random.sample(range(0,len(files)),len(files))
    dataset[i,:,:] = np.float32(pd.read_csv(path+"/"+file_name,header=None))
    print("\r", end="")
    print("import dataset progress: {}/{}: ".format(i+1,dataset_length), "▋" * int((i // (dataset_length/100))), end="")
    sys.stdout.flush()
print("\n")

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
scaler = MinMaxScaler(feature_range=(0, 1))


def get_model(width=128, height=128): #3.66
    """Build a 2D convolutional neural network model."""
    inputs = tf.keras.Input((width, height, 1))
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=16, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    #  outputs = tf.keras.layers.Dense(units=1, activation="relu")(x)
    outputs = tf.keras.layers.Dense(units=4, activation="sigmoid")(x)
    # Define the model.
    model = keras.Model(inputs, outputs, name="2dcnn")
    return model

test_loss = []
train_loss = []
test_acc = []
train_acc = []
test_auc = []
train_auc = []
y = np.array(y)
random.seed(13)
random_index = random.sample(range(0,dataset_length),dataset_length)
y = y[random_index]
dataset =dataset[random_index,:,:]
k_i = 0

for train_index, test_index in kfold.split(dataset):
    print(k_i)
    if k_i != K:
        k_i+=1
        continue
    k_i += 1
    print('train: %s, test: %s' % (train_index, test_index))
    X_train = dataset[:,:,train_index]
    y_train = y[train_index]
    X_test = dataset[test_index]
    y_test = y[test_index]
    X_train = np.expand_dims(X_train, 3)
    X_test = np.expand_dims(X_test, 3)
    y_train = np.expand_dims(y_train, 1)
    y_test = np.expand_dims(y_test, 1)
    # X_train = np.reshape(X_train, (X_train.shape[0], 600,60))
    # X_test = np.reshape(X_test, (X_test.shape[0], 600,60))

    # y_train = pd.DataFrame(data=y_train)
    # y_test = pd.DataFrame(data=y_test)
    one_hot_train_labels = to_categorical(y_train)
    one_hot_test_labels = to_categorical(y_test)
    #  Build model.
    model = get_model(width=2049, height=120)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy','AUC'])
    history = model.fit(X_train, one_hot_train_labels, epochs=500, batch_size=32, validation_data=(X_test, one_hot_test_labels))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_loss.append(loss)
    test_loss.append(val_loss)
    train_acc_1 = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    test_acc.append(val_acc)
    train_acc.append(train_acc_1)
    a = history.history['auc']
    b = history.history['val_auc']
    test_auc.append(b)
    train_auc.append(a)


model.save('content/output/Fourier_2D_model_'+str(K)+'.h5')
np.save('content/output/Fourier_2D_train_loss_'+str(K)+'.npy',train_loss)
np.save('content/output/Fourier_2D_test_loss_'+str(K)+'.npy',test_loss)
np.save('content/output/Fourier_2D_train_acc_'+str(K)+'.npy',train_acc)
np.save('content/output/Fourier_2D_test_acc_'+str(K)+'.npy',test_acc)
np.save('content/output/Fourier_2D_train_auc_'+str(K)+'.npy',train_auc)
np.save('content/output/Fourier_2D_test_auc_'+str(K)+'.npy',test_auc)