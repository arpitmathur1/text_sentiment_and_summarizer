import nltk
from nltk.corpus import stopwords
import pandas as pd
from gensim.models import Word2Vec

from keras.models import *
from keras.layers import *
from keras.callbacks import *

from torchtext.vocab import GloVe

import random
random.seed(0)


train = pd.read_csv("news_summary_more.csv")
print("before removing duplicates", len(train))

# drop some duplicates
train = train.drop_duplicates(subset='text', keep="last")

print("after removing duplicates", len(train))
train = train.reset_index(drop=True)
print(train.head())

text = "hi my name is Voldemort"

sentences = nltk.word_tokenize(text)


wordVectors = GloVe(name='6B', dim=100)


model=Sequential()

#embedding layer
model.add(Embedding(size_of_vocabulary,300,input_length=100,trainable=True)) 

#lstm layer
model.add(LSTM(128,return_sequences=True,dropout=0.2))

#Global Maxpooling
model.add(GlobalMaxPooling1D())

#Dense Layer
model.add(Dense(64,activation='relu')) 
model.add(Dense(1,activation='sigmoid')) 

#Add loss function, metrics, optimizer
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=["acc"]) 

#Adding callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)  
mc=ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True,verbose=1)  

#Print summary of model
print(model.summary())