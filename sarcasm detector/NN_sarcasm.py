# -*- coding: utf-8 -*-
"""
Created on Tue May 18 19:00:11 2021

@author: Administrator
"""

# keras for deep learning model creation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D
from keras.models import Model
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import tensorflow as tf
import re
#%%
train_data=pd.read_csv("c:/temp/train_data.csv",index_col=('Unnamed: 0'))
embedding_matrix=pd.read_csv('c:/temp/embedding_matrix.csv',index_col=('Unnamed: 0'))
target=pd.read_csv('c:/temp/target.csv',index_col=('Unnamed: 0'))
input_layer=Input(shape=(50,))
embedding_layer= Embedding(40000,300,weights=[embedding_matrix])(input_layer)

LSTM_layer = Bidirectional(LSTM(128, return_sequences = True))(embedding_layer)
maxpool_layer = GlobalMaxPool1D()(LSTM_layer)

dense_layer_1 = Dense(64, activation="relu")(maxpool_layer)
dropout_1 = Dropout(0.5)(dense_layer_1)

dense_layer_2 = Dense(32, activation="relu")(dropout_1)
dropout_2 = Dropout(0.5)(dense_layer_2)

output_layer = Dense(1, activation="sigmoid")(dropout_2)

model = Model(input_layer,output_layer)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#%%
model.fit(train_data,target,epochs=4,batch_size=512)
#%%
model.evaluate(train_data,target)