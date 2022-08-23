import random

K_index = 0

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

import mat73


df_backup = pd.read_csv(r'content/CNN_df.csv')
df = df_backup.copy()
print(df.shape)

df = df.sample(frac=1)
df['Stages'] = df['Stages'].astype(int)
df['Stages'].value_counts()

colors = ['#A6BEF3', '#9FCF9D', '#F5EBBC', '#F5B8B4']

plt.figure(figsize=(6.5,5.5))
g=sns.scatterplot(
    x="x", y="y",
    hue="Stages",
    # palette=['#6E9ECE', '#E6928F', '#84574D', '#76ba80'],
    palette=colors,
    # palette="coolwarm",
    data=df,
    legend=False,
    size="Correctness",
    sizes=(130, 80),
    style="Correctness",
    # edgecolor=['grey'],
    # alpha=1,
    # linewidth=0.3

)

g=sns.scatterplot(
    x="x", y="y",
    hue="Stages",
    # palette=['#6E9ECE', '#E6928F', '#84574D', '#76ba80'],
    palette=colors,
    # palette="coolwarm",
    data=df,
    # legend=False,
    size="Correctness",
    sizes=(130, 80),
    style="Correctness",
    edgecolor='#6F6F6F',
    alpha=1,
    linewidth=0.5
)


# Set x-axis label
plt.xlabel('t-SNE dimension 1', fontsize=23)
# Set y-axis label
plt.ylabel('t-SNE dimension 2', fontsize=23)
plt.title("STFT 2D-CNN test sample data", fontsize=23)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
# Set the legend
handles, labels = g.get_legend_handles_labels()
for h in handles[-2:]:
  h.set_facecolor('none')
  h.set_sizes([50])
plt.legend(fontsize=15, loc = 2, bbox_to_anchor = (1,1.01), markerscale=2.1, frameon=False)
plt.savefig('content\CNN test sample data_1')
plt.show()
#@title
plt.figure(figsize=(6.5,5.5))
g=sns.scatterplot(
    x="x", y="y",
    hue="Stages",
    # palette=['#6E9ECE', '#E6928F', '#84574D', '#76ba80'],
    palette="coolwarm",
    data=df,
    legend=False,
    size="Correctness",
    sizes=(130, 80),
    style="Correctness",
    # edgecolor=['grey'],
    # alpha=1,
    # linewidth=0.3

)

g=sns.scatterplot(
    x="x", y="y",
    hue="Stages",
    # palette=['#6E9ECE', '#E6928F', '#84574D', '#76ba80'],
    palette="coolwarm",
    data=df,
    legend=False,
    size="Correctness",
    sizes=(130, 80),
    style="Correctness",
    edgecolor='#6F6F6F',
    alpha=1,
    linewidth=0.5
)


# Set x-axis label
plt.xlabel('t-SNE dimension 1', fontsize=23)
# Set y-axis label
plt.ylabel('t-SNE dimension 2', fontsize=23)
plt.title("STFT 2D-CNN test sample data", fontsize=23)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.savefig('content\CNN test sample data_2')
plt.show()
