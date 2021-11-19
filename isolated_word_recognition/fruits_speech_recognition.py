#!/usr/bin/env python
# coding: utf-8
import glob
import time
import numpy as np
from hmmlearn import hmm
from scipy.io import wavfile
from collections import defaultdict
from python_speech_features import mfcc
import concurrent.futures as cf

start_time = time.perf_counter()

db = defaultdict(list)

for path in glob.glob('datasets/fruits/*'):
    label = path.split('\\')[-1]
    for wav_path in glob.glob(path + '/*'):
        sr, signal = wavfile.read(wav_path)
        db[label].append(signal.astype('float'))

labels = db.keys()

# get features
# 1. pre-emphasis with 0.97
# 2. framing with each frame size 25ms, step size 10ms
# 3. apply Hamming window function
# 4. FFT
# 5. Mel frequency cepstrum

def get_mfcc(x):
    return mfcc(x, sr, winfunc=np.hamming)

def get_list_mfcc(label):
    with cf.ThreadPoolExecutor() as executor:
        results = executor.map(get_mfcc, db[label])
    return list(results)


with cf.ThreadPoolExecutor() as executor:
    features = executor.map(get_list_mfcc, labels)
    features = dict(zip(labels, features))

train = dict()
test = dict()
test_size = int(0.2 * len(features['apple']))
for label in features.keys():
    train[label] = np.array([])
    test[label] = list()
    for feature in features[label][:-test_size]:
        if train[label].shape[0] == 0:
            train[label] = feature
        else:
            train[label] = np.append(train[label], feature, axis=0)
    for feature in features[label][-test_size:]:
        test[label].append(feature)


def train_hmm(label):
    model = hmm.GaussianHMM(n_components=5, n_iter=100)
    model.fit(train[label])
    return model

with cf.ThreadPoolExecutor() as executor:
    models = executor.map(train_hmm, labels, chunksize=10)
    models = dict(zip(labels, models))

def predict_speech(sample):
    def get_ll(model):
        ll, hidden = model.decode(sample)
        return ll
    predicted_ll = [get_ll(model) for model in models.values()]
    idx = np.argmax(predicted_ll)
    return list(labels)[idx]

len_data = 0
right_predicted = 0
for label, data in test.items():
    for datum in data:
        pred = predict_speech(datum)
        len_data += 1
        right_predicted += pred == label
print("Accuracy: %.2f" % (right_predicted / len_data))