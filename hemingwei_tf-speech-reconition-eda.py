#!/usr/bin/env python
# coding: utf-8



import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd




# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa
from sklearn.decomposition import PCA




# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
get_ipython().run_line_magic('matplotlib', 'inline')




train_audio_path = '/kaggle/input/tensorflow-speech-recognition-challenge/train/audio'
filename = 'yes/0a7c2a8d_nohash_0.wav'
sample_rate, samples = wavfile.read(join(train_audio_path, filename))




ll /kaggle/input/tensorflow-speech-recognition-challenge/train/audio/yes/0a7c2a8d_nohash_0.wav -h




sample_rate, samples




# define a function that calculates spectrogram
def log_specgram(audio, smaple_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio, fs=sample_rate, window='hann',         nperseg=nperseg, noverlap=noverlap, detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)




freqs, times, spec = log_specgram(samples, sample_rate)




freqs.shape, times.shape, spec.shape




fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + filename)
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

ax2 = fig.add_subplot(212)
ax2.imshow(spec.T, aspect='auto', origin='lower',     extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ' + filename)
ax2.set_ylabel('Freqs in HZ')
ax2.set_xlabel('Seconds')




# input normalization for NN
mean = np.mean(spec, axis=0)
std = np.std(spec, axis=0)
spectrogram = (spec - mean) / std




spectrogram.shape




sample_rate




# MFCC 梅尔频率倒谱系数
S = librosa.feature.melspectrogram(samples.astype(float), sr=sample_rate, n_mels=128)
# Convert to log scale (dB). we'll use peak power (max) as reference
log_S = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(12, 4))
librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
plt.title('Mel power spectrogram')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()




mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)
mfcc.shape, delta2_mfcc.shape




plt.figure(figsize=(12, 4))
librosa.display.specshow(delta2_mfcc)
plt.ylabel('MFCC coeffs')
plt.xlabel('Time')
plt.title('MFCC')
plt.colorbar()
plt.tight_layout()




# Spectrogram in 3d
data = [go.Surface(x=times, y=freqs, z=spectrogram.T)]
layout = go.Layout(
    title='Spectrogram of "yes" in 3d',
    scene = dict(
        yaxis = dict(title='Frequencies', range=[freqs.min(), freqs.max()]),
        xaxis = dict(title='Time', range=[times.min(), times.max()]),
        zaxis = dict(title='Log amplitude')
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)




# Silence removal
print(samples.shape)
ipd.Audio(samples, rate=sample_rate)




sample_cut = samples[4000:13000]
print(sample_cut.shape)
ipd.Audio(sample_cut, rate=sample_rate)




# guessed slignment of each letter in 'yes'
freqs, times, spectrogram_cut = log_specgram(sample_cut, sample_rate)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + filename)
ax1.set_ylabel('Amplitude')
ax1.plot(sample_cut)

ax2 = fig.add_subplot(212)
ax2.set_title('Spectrogram of ' + filename)
ax2.set_ylabel('Frequencies * 0.1')
ax2.set_xlabel('Samples')
ax2.imshow(spectrogram_cut.T, aspect='auto', origin='lower',
          extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.text(0.06, 1000, 'Y', fontsize=18)
ax2.text(0.17, 1000, 'E', fontsize=18)
ax2.text(0.36, 1000, 'S', fontsize=18)
xcoords = [0.025, 0.11, 0.23, 0.49]
for xc in xcoords:
    ax1.axvline(x=xc*16000, c='r')
    ax2.axvline(x=xc, c='r')




# Resampling - dimension reduction
filename = 'happy/0b09edd3_nohash_0.wav'
new_sample_rate = 8000
sample_rate, samples = wavfile.read(join(train_audio_path, filename))
resampled = signal.resample(samples, int(new_sample_rate/sample_rate*samples.shape[0]))




int(new_sample_rate/sample_rate*samples.shape[0])




ipd.Audio(samples, rate=sample_rate)




ipd.Audio(resampled, rate=new_sample_rate)




# Fast fourier transform
def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    vals = 2.0 / N * np.abs(yf[0:N//2])
    return xf, vals




xf, vals = custom_fft(samples, sample_rate)
plt.figure(figsize=(12, 4))
plt.title('FFT of recording sampled with ' + str(sample_rate) + ' Hz')
plt.plot(xf, vals)
plt.xlabel('Frequency')
plt.grid()
plt.show()




xf, vals = custom_fft(resampled, new_sample_rate)
plt.figure(figsize=(12, 4))
plt.title('FFT of recording sampled with ' + str(new_sample_rate) + ' Hz')
plt.plot(xf, vals)
plt.xlabel('Frequency')
plt.grid()
plt.show()




# number of records
dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
dirs.sort()
print('Number of labels: ' + str(len(dirs)))




# calculate
number_of_recordings = []
for direct in dirs:
    waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]
    number_of_recordings.append(len(waves))




# number of total recordings in train sets
sum(number_of_recordings)




# plot
data = [go.Histogram(x=dirs, y=number_of_recordings)]
trace = go.Bar(
    x=dirs,
    y=number_of_recordings,
    marker=dict(color=number_of_recordings, colorscale='viridis', showscale=True)
)
layout=go.Layout(
    title='Number of recordings in given label',
    xaxis=dict(title='Words'),
    yaxis=dict(title='Number of recordings')
)
py.iplot(go.Figure(data=[trace], layout=layout))




# Speaker doesn't occur in both train and test data sets
filenames = ['on/004ae714_nohash_0.wav', 'on/0137b3f4_nohash_0.wav']
for filename in filenames:
    sample_rate, samples = wavfile.read(join(train_audio_path, filename))
    xf, vals = custom_fft(samples, sample_rate)
    plt.figure(figsize=(12, 4))
    plt.title('FFT of speaker ' + filename[4:11])
    plt.plot(xf, vals)
    plt.xlabel('Frequency')
    plt.grid()
    plt.show()




print('Speaker ' + filenames[0][4:11])
ipd.Audio(join(train_audio_path, filenames[0]))




print('Speaker ' + filenames[1][4:11])
ipd.Audio(join(train_audio_path, filenames[1]))




# Recordings with some weird silence
filename = 'yes/01bb6a2a_nohash_1.wav'
sample_rate, samples = wavfile.read(join(train_audio_path, filename))
freqs, times, spectrogram = log_specgram(samples, sample_rate)
plt.figure(figsize=(10, 7))
plt.title('Spectrogram of ' + filename)
plt.ylabel('Freqs')
plt.xlabel('Times')
plt.imshow(spectrogram.T, aspect='auto', origin='lower',     extent=[times.min(), times.max(), freqs.min(), freqs.max()])
plt.yticks(freqs[::16])
plt.xticks(times[::16])
plt.show()




ipd.Audio(join(train_audio_path, filename))




# Calculating number of recordings shorter than 1 second
num_of_shorter = 0
for d in dirs:
    waves = [f for f in os.listdir(join(train_audio_path, d)) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(join(train_audio_path, d, wav))
        if samples.shape[0] < sample_rate:
            num_of_shorter += 1
print('Number of recordings shorter than 1 second: ' + str(num_of_shorter))




# Mean spectrograms and FFT for each word
to_keep = 'yes no up down left right on off stop go'.split()
dirs = [d for d in dirs if d in to_keep]
print(dirs)
for d in dirs:
    vals_all = []
    spec_all = []
    waves = [f for f in os.listdir(join(train_audio_path, d)) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(join(train_audio_path, d, wav))
        if samples.shape[0] != 16000: continue
        xf, vals = custom_fft(samples, 16000)
        vals_all.append(vals)
        freqs, times, spec = log_specgram(samples, 16000)
        spec_all.append(spec)
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.title('Mean fft of ' + d)
    plt.plot(np.mean(np.array(vals_all), axis=0))
    plt.grid()
    plt.subplot(122)
    plt.title('Mean spectrogram of ' + d)
    plt.imshow(np.mean(np.array(spec_all), axis=0).T, aspect='auto', origin='lower',         extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    plt.yticks(freqs[::16])
    plt.xticks(times[::16])
    plt.show()




# Frequenciy components across the words
def violinplot_frequency(dirs, freq_ind):
    spec_all = []
    for idx, d in enumerate(dirs):
        spec_all.append([])
        waves = [f for f in os.listdir(join(train_audio_path, d)) if f.endswith('.wav')]
        for wav in waves[:100]:
            sample_rate, samples = wavfile.read(join(train_audio_path, d, wav))
            freqs, times, spec = log_specgram(samples, sample_rate)
            spec_all[idx].extend(spec[:, freq_ind])
    minimum = min([len(spec) for spec in spec_all])
    spec_all = np.array([spec[:minimum] for spec in spec_all])
    plt.figure(figsize=(13, 7))
    plt.title('Frequency ' + str(freqs[freq_ind]) + ' Hz')
    plt.ylabel('Amount of frequency in a word')
    plt.xlabel('Words')
    sns.violinplot(data=pd.DataFrame(spec_all.T, columns=dirs))
    plt.show()      




violinplot_frequency(dirs, 20)




violinplot_frequency(dirs, 50)




violinplot_frequency(dirs, 120)




# Anomaly detection
fft_all = []
names = []
for d in dirs:
    waves = [f for f in os.listdir(join(train_audio_path, d)) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(join(train_audio_path, d, wav))
        if samples.shape[0] != sample_rate:
            samples = np.append(samples, np.zeros((sample_rate - samples.shape[0],)))
        x, val = custom_fft(samples, sample_rate)
        fft_all.append(val)
        names.append(join(d, wav))
fft_all = np.array(fft_all)

# Normalization
fft_all = (fft_all - np.mean(fft_all, axis=0)) / np.std(fft_all, axis=0)

# Dimension reduction
pca = PCA(n_components=3)
fft_all = pca.fit_transform(fft_all)

def interactive_3d_plot(data, names):
    scatt = go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], mode='markers', text=names)
    data = go.Data([scatt])
    layout = go.Layout(title='Anomaly detection')
    figure = go.Figure(data=data, layout=layout)
    py.iplot(figure)
interactive_3d_plot(fft_all, names)




len(fft_all)




# Anomaly samples
print('Recording go/0487ba9b_nohash_0.wav')
ipd.Audio(join(train_audio_path, 'go/0487ba9b_nohash_0.wav'))




print('Recording yes/e4b02540_nohash_0.wav')
ipd.Audio(join(train_audio_path, 'yes/e4b02540_nohash_0.wav'))




print('Recording seven/b1114e4f_nohash_0.wav')
ipd.Audio(join(train_audio_path, 'seven/b1114e4f_nohash_0.wav'))






