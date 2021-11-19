#!/usr/bin/env python
# coding: utf-8
import glob
import numpy as np
import seaborn as sns
from hmmlearn import hmm
from scipy.io import wavfile
import matplotlib.pyplot as plt
from collections import defaultdict
from python_speech_features import mfcc
import concurrent.futures as cf

data_path = glob.glob('datasets/digits/*')

sr, _ = wavfile.read(data_path[0])

labels = np.arange(10).tolist()

db = defaultdict(list)
for path in data_path:
    label = int(path.split('_', 1)[0][-1])
    _, signal = wavfile.read(path)
    db[label].append(signal)


def feature_analysis(x):
    return mfcc(x, sr, winlen=0.025, winstep=0.01, preemph=0.95, winfunc=np.hamming)


def get_list_mfcc(label):
    with cf.ThreadPoolExecutor() as executor:
        data = executor.map(feature_analysis, db[label])
    return list(data)


with cf.ThreadPoolExecutor() as executor:
    data = executor.map(get_list_mfcc, labels)
    data = dict(zip(labels, data))


all_size = len(data[0])

train_size = int(0.8 * all_size)
test_size = all_size - train_size

train = dict()
test = dict()
for label in labels:
    train[label] = np.concatenate(data[label][:train_size], axis=0)
    test[label] = data[label][-test_size:]


def loglh(model, observation):
    ll, _ = model.decode(observation)
    return ll


def predict_speech(observation):
    loglikelihood = [loglh(model, observation)
                     for model in speech_models.values()]
    return np.argmax(loglikelihood)


def train_hmm(label):
    model = hmm.GMMHMM(n_components=2, n_mix=2, n_iter=1000, verbose=False)
    model.fit(train[label])
    return model


with cf.ThreadPoolExecutor() as executor:
    speech_models = executor.map(train_hmm, labels, chunksize=3)
    speech_models = dict(zip(labels, speech_models))

true = 0
test_data_size = 0
confusion_matrix = np.zeros(shape=(10, 10))
for label in range(10):
    for observation in test[label]:
        pred = predict_speech(observation)
        true += pred == label
        confusion_matrix[label, pred] += 1
        test_data_size += 1
print("Accuracy: %.2f" % (true / test_data_size))
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix, annot=True, fmt='.0f', ax=ax)
# ax.set_xlabel('')
plt.savefig('confusion_matrix.png')
