# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import collections


os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
number_of_replica = 3
increment_size = 500

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

data_train = pd.read_csv('/media/nahid/Windows8_OS/labeledTrainData/labeledTrainData.tsv', sep='\t')
print data_train.shape

texts = []
labels = []

for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx])
    texts.append(clean_str(text.get_text().encode('ascii','ignore')))
    labels.append(data_train.sentiment[idx])


print "Using TF-IDF"
vectorizer = TfidfVectorizer(analyzer="word", \
                             tokenizer=None, \
                             preprocessor=None, \
                             stop_words=None, \
                             max_features=15000)

data = vectorizer.fit_transform(texts)
data = data.toarray()
print data.shape

indices = np.arange(data.shape[0])
np.random.seed(seed=100)
np.random.shuffle(indices)
data = data[indices]
print type(indices)
label = []
for index in indices:
    label.append(labels[index])

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = label[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = label[-nb_validation_samples:]


x_train_incremental = x_train[:increment_size]
y_train_incremental = y_train[:increment_size]
model = LogisticRegression()
model.fit(x_train_incremental, y_train_incremental)


s = ""

x_train_incremental = x_train_incremental.tolist()
initial_len = len(x_train_incremental)
for replicate in xrange(0, number_of_replica - 1):
    for index in xrange(0, initial_len):
        x_train_incremental.append(x_train_incremental[index])
        y_train_incremental.append(y_train_incremental[index])

current_size = initial_len

while current_size <= len(y_train) - increment_size:
    print "train size", len(y_train_incremental)

    y_pred = model.predict(x_val)
    f1score = f1_score(y_val, y_pred, average='binary')
    acc = accuracy_score(y_val, y_pred)*100

    s = s + str(acc) + ","
    print s

    print "Logistic Regression - Accuracy:", acc, "f1score", f1score
    current_size = len(y_train_incremental)  -  (number_of_replica - 1)*initial_len
    if current_size == len(y_train):
        break


    x_train_next_batch = x_train[current_size:current_size+increment_size]

    for train_x in x_train_next_batch:
        x_train_incremental.append(train_x)

    y_pred_incremental = model.predict(x_train_next_batch)
    for y_pred_i in y_pred_incremental:
        y_train_incremental.append(y_pred_i)

    model = LogisticRegression()
    model.fit(x_train_incremental, y_train_incremental)
    current_size = current_size + increment_size


print "train size after all exhausted:", len(y_train_incremental)
y_pred = model.predict(x_val)
f1score = f1_score(y_val, y_pred, average='binary')
acc = accuracy_score(y_val, y_pred)*100
print "Logistic Regression - Accuracy:", acc, "f1score", f1score
s= s+str(acc)+","

print s

text_file = open("result_Logit_batch_"+ str(increment_size)+"_replica_"+str(number_of_replica) + ".txt", "w")
text_file.write(s)
text_file.close()

exit(0)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_val)
f1score = f1_score(y_val, y_pred, average='binary')
acc = accuracy_score(y_val, y_pred)

print "Logistic Regression - Accuracy:", acc, "f1score", f1score

data_unlabeled = pd.read_csv('/media/nahid/Windows8_OS/unlabeledTrainData/unlabeledTrainData.tsv', sep='\t')
print data_unlabeled.shape
texts_unlabeled = []

for idx in range(data_unlabeled.review.shape[0]):
    text = BeautifulSoup(data_unlabeled.review[idx])
    texts_unlabeled.append(clean_str(text.get_text().encode('ascii','ignore')))


print "Using TF-IDF"
vectorizer_unlabeled = TfidfVectorizer(analyzer="word", \
                             tokenizer=None, \
                             preprocessor=None, \
                             stop_words=None, \
                             max_features=15000)

data_unlabeled_tfidf = vectorizer_unlabeled.fit_transform(texts_unlabeled)
data_unlabeled_tfidf = data_unlabeled_tfidf.toarray()
print data_unlabeled_tfidf.shape

y_pred_unlabeled = model.predict(data_unlabeled_tfidf)
print type(y_pred_unlabeled)
#y_pred_unlabeled = np.asarray(y_pred_unlabeled,dtype=np.int32)
y_pred_unlabeled = y_pred_unlabeled.astype(int)
print type(y_pred_unlabeled)
np.savetxt('test.out', y_pred_unlabeled, delimiter=',')