#!/usr/bin/env python
# coding: utf-8
import glob
import numpy as np
from hmmlearn import hmm
from scipy.io import wavfile
from collections import defaultdict
from python_speech_features import mfcc

db = defaultdict(list)

for path in glob.glob('datasets/fruits/*'):
    label = path.split('\\')[-1]
    for wav_path in glob.glob(path + '/*'):
        sr, signal = wavfile.read(wav_path)
        db[label].append(signal.astype('float'))


# get features
# 1. pre-emphasis with 0.97
# 2. framing with each frame size 25ms, step size 10ms
# 3. apply Hamming window function
# 4. FFT
# 5. Mel frequency cepstrum
features = defaultdict(list)
for label in db.keys():
    features[label] = list(
        map(lambda x: mfcc(x, sr, winfunc=np.hamming), db[label]))


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


models = dict()
for label in features.keys():
    models[label] = hmm.GaussianHMM(n_components=5, n_iter=100)
    models[label].fit(train[label])


def predict_speech(sample):
    def get_ll(model):
        ll, hidden = model.decode(sample)
        return ll
    predicted_ll = [get_ll(model) for label, model in models.items()]
    labels = list(models.keys())
    idx = np.argmax(predicted_ll)
    return labels[idx]

len_data = 0
right_predicted = 0
for label, data in test.items():
    for datum in data:
        pred = predict_speech(datum)
        len_data += 1
        right_predicted += pred == label
#         print("Actual: {}, Predicted: {}".format(label, pred))
print("Accuracy: %.2f" % (right_predicted / len_data))
