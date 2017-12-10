# author - Md Mustafizur Rahman
# Dec 08, 2017
import numpy as np
import pandas as pd
import re

from bs4 import BeautifulSoup
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import Queue
from math import log
import pickle
import time
import sys

class relevance(object):
    def __init__(self, priority, index):
        self.priority = priority
        self.index = index
        return
    def __cmp__(self, other):
        return -cmp(self.priority, other.priority)


'''
active_sampling = True
number_of_replica = 10
seed_size = 100
increment_size = 50
'''

active_sampling = True if sys.argv[1] == "True" else False
number_of_replica = int(sys.argv[2])
seed_size = int(sys.argv[3])
increment_size = int(sys.argv[4])

print active_sampling, str(number_of_replica), str(seed_size), str(increment_size)


classifier_type = 1

classifier_name = "LR"
pre_loaded = True
sampling_type = ""
base_address = "/media/nahid/Windows8_OS/data2/"
result_address = base_address + "result/"

os.environ['KERAS_BACKEND']='theano'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2



if int(classifier_type) == 1:
    classifier_name = "LR"

if active_sampling == False:
    sampling_type = "Random"
else:
    sampling_type = "Active"
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

    #x_train = data[:-nb_validation_samples]
    #y_train = label[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = label[-nb_validation_samples:]

    # loading and using 20,000 training is not helping so loading only 5,000
    x_train = data[:5000]
    y_train = label[:5000]

    output = open(base_address+"x_train.txt", 'wb')
    pickle.dump(x_train, output)
    output.close()

    output = open(base_address+"x_val.txt", 'wb')
    pickle.dump(x_val, output)
    output.close()

    output = open(base_address+"y_train.txt", 'wb')
    pickle.dump(y_train, output)
    output.close()

    output = open(base_address+"y_val.txt", 'wb')
    pickle.dump(y_val, output)
    output.close()



else:
    print "Pre-loading is Active"
    start_time = time.time()
    input = open(base_address+"x_train.txt", 'rb')
    x_train = pickle.load(input)
    print("--- %s seconds ---" % (time.time() - start_time))

    input.close()

    start_time = time.time()
    input = open(base_address+"x_val.txt", 'rb')
    x_val = pickle.load(input)
    input.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    input = open(base_address+"y_train.txt", 'rb')
    y_train = pickle.load(input)
    input.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    input = open(base_address+"y_val.txt", 'rb')
    y_val = pickle.load(input)
    input.close()
    print("--- %s seconds ---" % (time.time() - start_time))

x_train_incremental = x_train[:seed_size]
y_train_incremental = y_train[:seed_size]
model = LogisticRegression()
model.fit(x_train_incremental, y_train_incremental)

s = ""
x_train_incremental = x_train_incremental.tolist()

x_train_list = [] # keep the index of the X which is added in the train set
for index in xrange(0, seed_size):
    x_train_list.append(index)

initial_len = len(x_train_incremental)

for replicate in xrange(0, number_of_replica - 1):
    for index in xrange(0, initial_len):
        x_train_incremental.append(x_train_incremental[index])
        y_train_incremental.append(y_train_incremental[index])

current_size = initial_len
while current_size <= len(y_train) - increment_size:
    print "train size", current_size

    y_pred = model.predict(x_val)
    f1score = f1_score(y_val, y_pred, average='binary')
    acc = accuracy_score(y_val, y_pred)*100

    s = s + str(acc) + ","
    print s

    print "Logistic Regression - Accuracy:", acc, "f1score", f1score
    current_size = len(y_train_incremental)  -  (number_of_replica - 1)*initial_len

    if current_size == len(y_train):
        break

    if active_sampling == False: # random sampling
        x_train_next_batch = x_train[current_size:current_size+increment_size]

        for train_x in x_train_next_batch:
            x_train_incremental.append(train_x)

        y_pred_incremental = model.predict(x_train_next_batch)
        for y_pred_i in y_pred_incremental:
            y_train_incremental.append(y_pred_i)
    else:
        x_predict_list = []  # keep the element for prediction
        y_predicted_list = {}
        queueSize = len(x_train) - len(x_train_list)
        queue = Queue.PriorityQueue(queueSize)

        for index in xrange(0, len(x_train)):
            if index not in x_train_list:
                y_prob = model.predict_proba(np.array(x_train[index]).reshape(1, -1))[0]
                y_predicted_list[index] = model.predict(np.array(x_train[index]).reshape(1, -1))[0]
                entropy = (-1) * (y_prob[0] * log(y_prob[0], 2) + y_prob[1] * log(y_prob[1], 2))
                queue.put(relevance(entropy, index))
                # queue.put(relevance(y_prob[1], index))

        counter = 0
        while not queue.empty():
            if counter == increment_size:
                break
            item = queue.get()
            index = item.index
            y_train_incremental.append(y_predicted_list[index])
            x_train_incremental.append(x_train[index])
            x_train_list.append(index)
            counter = counter + 1

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

text_file = open(result_address+"classifier_" +classifier_name+ "_selection_"+ sampling_type +"_seed_"+ str(seed_size)+ "_batch_"+str(increment_size)+"_replica_"+str(number_of_replica) + ".txt", "w")
text_file.write(s)
text_file.close()

exit(0)
#############################
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