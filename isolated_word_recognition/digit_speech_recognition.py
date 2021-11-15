#!/usr/bin/env python
# coding: utf-8
import glob
import numpy as np
from hmmlearn import hmm
from scipy.io import wavfile
from collections import defaultdict
from python_speech_features import mfcc


data_path = glob.glob('datasets/digits/*')


def feature_analysis(x):
    return mfcc(x, sr, winlen=0.025, winstep=0.01, preemph=0.95, winfunc=np.hamming)


data = defaultdict(list)
for path in data_path:
    sr, signal = wavfile.read(path)
    # get features from speech signal
    signal = feature_analysis(signal)
    label = path.split('\\')[-1].split('_', 1)[0]
    data[int(label)].append(signal)


train = dict()
test = dict()

all_size = len(data[0])

train_size = int(0.8 * all_size)
test_size = int(all_size - train_size)

print(train_size, test_size)

for label in data.keys():
    train[label] = np.array([])
    test[label] = list()
    for feature in data[label][:train_size]:
        if train[label].shape[0] == 0:
            train[label] = feature
        else:
            train[label] = np.append(train[label], feature, axis=0)
    for feature in data[label][-test_size:]:
        test[label].append(feature)


def loglh(model, observation):
    ll, _ = model.decode(observation)
    return ll


def predict_speech(observation):
    loglikelihood = [loglh(model, observation)
                     for model in speech_model.values()]
    return np.argmax(loglikelihood)


speech_model = dict()
for label in np.arange(10):
    speech_model[label] = hmm.GMMHMM(
        n_components=2, n_mix=2, n_iter=1000, verbose=False)
    speech_model[label].fit(train[label])


true = 0
test_data_size = 0
for label in range(10):
    for observation in test[label]:
        pred = predict_speech(observation)
        true += pred == label
        test_data_size += 1
print("Accuracy: %.2f" % (true / test_data_size))
