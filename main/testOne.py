import os
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import tqdm
import warnings

from keras.models import load_model
model = load_model('C:\\Users\Dmitriy\\eclipse-workspace\\HSENeuralNetworksSpeachRecognition\\main\\speech2text_model.hdf5')

# Subsample
all_label = [
    'left', 'go', 'yes', 'down', 'up', 'on', 'right', 'no', 'off', 'stop',
]

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

label_enconder = LabelEncoder()
y = label_enconder.fit_transform(all_label)
classes = list(label_enconder.classes_)

def s2t_predict(audio, shape_num=8000):
    prob=model.predict(audio.reshape(1,shape_num,1))
    index=np.argmax(prob[0])
    return classes[index]

samples, sample_rate = librosa.load('C:\\SpeechDataset\\mytest\\record(4).wav', sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)

samples = samples[:8000]

ipd.Audio(samples, rate=8000)

samples.reshape(1,8000,1)

print(samples.shape)

print("Text:",s2t_predict(samples, 8000))

