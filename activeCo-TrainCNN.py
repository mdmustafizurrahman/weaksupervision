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

os.environ['KERAS_BACKEND'] = 'theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model
import Queue
from math import log
from numpy import argmax
import pickle
import time
from copy import deepcopy
from sklearn.linear_model import LogisticRegression

'''
active_sampling = True
number_of_replica = 1
seed_size = 100
increment_size = 100
random_seed = 100
'''


active_sampling = True if sys.argv[1] == "True" else False
number_of_replica = int(sys.argv[2])
seed_size = int(sys.argv[3])
increment_size = int(sys.argv[4])
random_seed = int(sys.argv[5])

print active_sampling, str(number_of_replica), str(seed_size), str(increment_size)


classifier_type = 2
classifier_name = "CNN"
pre_loaded = True
sampling_type = ""
base_address = "/media/nahid/Windows8_OS/coTrain_Large/data_CNN/"
result_address = base_address + "result/"

GLOVE_DIR = "/media/nahid/Windows8_OS/glove.6B"
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
NUMBER_OF_EPOCHS = 10

if active_sampling == False:
    sampling_type = "Random"
else:
    sampling_type = "Active"

print "Sampling Type", sampling_type

class relevance(object):
    def __init__(self, priority, index):
        self.priority = priority
        self.index = index
        return
    def __cmp__(self, other):
        return -cmp(self.priority, other.priority)



# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


if pre_loaded == False:
    data_train = pd.read_csv('/media/nahid/Windows8_OS/labeledTrainData/labeledTrainData.tsv', sep='\t')
    print data_train.shape

    texts = []
    labels = []

    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.sentiment[idx])

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    output = open(base_address + "word_index.txt", 'wb')
    pickle.dump(word_index, output)
    output.close()

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    print "type(data.shape[0]:)", type(data.shape)

    indices = np.arange(data.shape[0])
    #np.random.seed(seed=100)
    #np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    #x_train = data[:-nb_validation_samples]
    #y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    # loading and using 20,000 training is not helping so loading only 5,000
    x_train = data[:10000]
    y_train = labels[:10000]

    output = open(base_address + "x_train.txt", 'wb')
    pickle.dump(x_train, output)
    output.close()

    output = open(base_address + "x_val.txt", 'wb')
    pickle.dump(x_val, output)
    output.close()

    output = open(base_address + "y_train.txt", 'wb')
    pickle.dump(y_train, output)
    output.close()

    output = open(base_address + "y_val.txt", 'wb')
    pickle.dump(y_val, output)
    output.close()
    exit(0)
else:
    print "Pre-loading is Active"
    print "loading data for CNN format"
    base_address = "/media/nahid/Windows8_OS/coTrain_Large/data_CNN/"
    indices = np.arange(5000)
    np.random.seed(seed=random_seed)
    np.random.shuffle(indices)

    start_time = time.time()
    input = open(base_address + "x_train.txt", 'rb')
    x_train = pickle.load(input)
    x_train = x_train[indices]

    print("--- %s seconds ---" % (time.time() - start_time))

    input.close()

    start_time = time.time()
    input = open(base_address + "x_val.txt", 'rb')
    x_val = pickle.load(input)
    input.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    input = open(base_address + "y_train.txt", 'rb')
    y_train_CNN = pickle.load(input)
    y_train_CNN = y_train_CNN[indices]

    input.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    input = open(base_address + "y_val.txt", 'rb')
    y_val = pickle.load(input)
    input.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    input = open(base_address + "word_index.txt", 'rb')
    word_index = pickle.load(input)
    input.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    print "loading data for Logit format"
    print "Pre-loading is Active"
    base_address = "/media/nahid/Windows8_OS/coTrain_Large/data_logit/"

    start_time = time.time()
    input = open(base_address + "x_train.txt", 'rb')
    x_train_logit = pickle.load(input)
    x_train_logit = x_train_logit[indices]
    print("--- %s seconds ---" % (time.time() - start_time))

    input.close()

    start_time = time.time()
    input = open(base_address + "x_val.txt", 'rb')
    x_val_logit = pickle.load(input)
    input.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    input = open(base_address + "y_train.txt", 'rb')
    y_train_logit_tmp = pickle.load(input)
    y_train_logit = []
    for ind in indices:
        y_train_logit.append(y_train_logit_tmp[ind])
    input.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    input = open(base_address + "y_val.txt", 'rb')
    y_val_logit = pickle.load(input)
    input.close()
    print("--- %s seconds ---" % (time.time() - start_time))


print('Number of positive and negative reviews in traing and validation set ')
print y_train_CNN.sum(axis=0)
print y_val.sum(axis=0)


### logistic regression part#######

# training on the seed set
x_train_logit_seed = x_train_logit[:seed_size]
y_train_logit_seed = y_train_logit[:seed_size]

model = LogisticRegression()
model.fit(x_train_logit_seed, y_train_logit_seed)

y_pred_logit = model.predict(x_train_logit[seed_size:])
y_pred_logit_val = model.predict(x_val_logit)

s = ""
acc = accuracy_metric(y_val_logit, y_pred_logit_val)
print "Logistic - Accuracy:", acc
s=s+str(acc)+','

y_pred_logit = to_categorical(np.asarray(y_pred_logit))
y_train_CNN_seed = y_train_CNN[:seed_size]
y_train =  y_train_CNN_seed.tolist() + y_pred_logit.tolist()

print type(y_train), len(y_train)

###################################


embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
l_flat = Flatten()(l_pool3)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - simplified convolutional neural network")
model.summary()

x_train_incremental = x_train[:seed_size]
y_train_incremental = y_train[:seed_size]

print "seed sum:", np.asarray(y_train_incremental).sum(axis = 0)
model.fit(x_train_incremental, y_train_incremental, nb_epoch=NUMBER_OF_EPOCHS, batch_size=128)

x_train_list = [] # keep the index of the X which is added in the train set
for index in xrange(0, seed_size):
    x_train_list.append(index)

initial_len = len(x_train_incremental)

x_train_incremental = x_train_incremental.tolist()
#y_train_incremental = y_train_incremental.tolist()

for replicate in xrange(0, number_of_replica - 1):
    for index in xrange(0, initial_len):
        x_train_incremental.append(x_train_incremental[index])
        y_train_incremental.append(y_train_incremental[index])

current_size = initial_len

y_val_categorical = y_val
y_val = []
for val_categorical in y_val_categorical:
    #print val_categorical
    y_val.append(argmax(val_categorical))


while current_size <= len(y_train) - increment_size:
    print "train size", current_size

    y_pred_categorical = model.predict(x_val)
    y_pred = []
    for pred_categorical in y_pred_categorical:
        y_pred.append(argmax(pred_categorical))

    #f1score = f1_score(y_val, y_pred, average='binary')
    acc = accuracy_metric(y_val, y_pred)
    s= s+str(acc)+","
    print s
    print "CNN - Accuracy:", acc

    current_size = len(y_train_incremental) -  (number_of_replica - 1)*initial_len
    if current_size == len(y_train):
        break


    if active_sampling == False: # random sampling
        print "Random Sampling"
        x_train_next_batch = x_train[current_size:current_size+increment_size]

        for train_x in x_train_next_batch:
            x_train_incremental.append(train_x)

        y_pred_incremental = model.predict(x_train_next_batch)
        for y_pred_i in y_pred_incremental:
            #print "Type: y_pred", type(y_pred)
            #print "Size: y_pred",y_pred)

            # converting soft-label to hard label
            #print "y_pred_i", y_pred_i

            max_index = argmax(y_pred_i)
            min_index = 1 - max_index

            y_pred_i[max_index] = 1
            y_pred_i[min_index] = 0
            #print "y_pred_i", y_pred_i

            y_train_incremental.append(y_pred_i)
    else:
        x_predict_list = [] # keep the element for prediction
        y_predicted_list = {}
        queueSize = len(x_train) - len(x_train_list)
        queue = Queue.PriorityQueue(queueSize)
        print "Active Sampling"
        for index in xrange(0, len(x_train)):
            if index not in x_train_list:
                y_prob = model.predict(np.array(x_train[index]).reshape(1, -1))[0]

                # converting soft-label to hard label
                y_pred_i = deepcopy(y_prob)
                max_index = argmax(y_pred_i)
                min_index = 1 - max_index

                y_pred_i[max_index] = 1
                y_pred_i[min_index] = 0
                #print "y_pred_i", y_pred_i, "y_prob:", y_prob
                y_predicted_list[index] = y_pred_i
                #entropy = (-1) * (y_prob[0] * log(y_prob[0], 2) + y_prob[1] * log(y_prob[1], 2))
                max = 0.0
                # most confident
                if y_prob[0]> y_prob[1]:
                    max = y_prob[0]
                else:
                    max = y_prob[1]
                queue.put(relevance(max, index))
                #queue.put(relevance(y_prob[1], index))

        counter = 0

        while not queue.empty():
            if counter == increment_size:
                break
            item = queue.get()
            index = item.index
            #self train

            #y_train_incremental.append(y_predicted_list[index])

            # co train
            y_train_incremental.append(y_train[index])
            x_train_incremental.append(x_train[index])
            x_train_list.append(index)
            counter = counter + 1


    model.fit(x_train_incremental, y_train_incremental, epochs=NUMBER_OF_EPOCHS, batch_size=128)
    current_size = current_size + increment_size


print "train size after all exhausted:", len(y_train_incremental)
y_pred_categorical = model.predict(x_val)
y_pred = []
for pred_categorical in y_pred_categorical:
    y_pred.append(argmax(pred_categorical))

#f1score = f1_score(y_val, y_pred, average='binary')
acc = accuracy_metric(y_val, y_pred)
print "CNN - Accuracy:", acc
s= s+str(acc)+","

print s

text_file = open(result_address+"classifier_" +classifier_name+ "_selection_"+ sampling_type +"_seed_"+ str(seed_size)+ "_batch_"+str(increment_size)+"_replica_"+str(number_of_replica)+"_random_"+str(random_seed) + ".txt", "w")
text_file.write(s)
text_file.close()


exit(0)
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=128)


'''
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

# applying a more complex convolutional approach
convs = []
filter_sizes = [3, 4, 5]

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    l_conv = Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)

l_merge = Merge(mode='concat', concat_axis=1)(convs)
l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(30)(l_cov2)
l_flat = Flatten()(l_pool2)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - more complex convolutional neural network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=20, batch_size=50)
'''