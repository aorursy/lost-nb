#!/usr/bin/env python
# coding: utf-8



import pandas as pd




import librosa
import numpy    as np

from scipy.fftpack         import fft
from scipy                 import signal
from scipy.io              import wavfile




import matplotlib.pyplot as plt
import pandas            as pd

import IPython.display   as ipd
import librosa.display

import plotly.offline    as py

py.init_notebook_mode(connected=True)

get_ipython().run_line_magic('matplotlib', 'inline')




trainAudioPath = '../input/train/audio/'




from IPython.display import Markdown, display
def printMarkdown(string):
    display(Markdown(string))




def plotRawWave(plotTitle, sampleRate, samples, figWidth=14, figHeight=4):
    plt.figure(figsize=(figWidth, figHeight))
    plt.plot(np.linspace(0, sampleRate/len(samples), sampleRate), samples)
    plt.title("Raw sound wave of " + plotTitle)
    plt.ylabel("Amplitude")
    plt.xlabel("Time [sec]")
    plt.show()  # force display while in for loop
    return None




def computeLogSpectrogram(audio, sampleRate, windowSize=20, stepSize=10, epsilon=1e-10):
    nperseg  = int(round(windowSize * sampleRate / 1000))
    noverlap = int(round(stepSize   * sampleRate / 1000))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sampleRate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + epsilon)




def plotLogSpectrogram(plotTitle, freqs, times, spectrogram, figWidth=14, figHeight=4):
    fig = plt.figure(figsize=(figWidth, figHeight))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower', 
               cmap="inferno",   #  default was "viridis"  (perceptually uniform)
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    plt.colorbar(pad=0.01)
    plt.title('Spectrogram of ' + plotTitle)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    fig.tight_layout()
    plt.show()  # force display while in for loop
    return None




def computeLogMelSpectrogram(samples, sampleRate, nMels=128):
    melSpectrum = librosa.feature.melspectrogram(samples, sr=sampleRate, n_mels=nMels)
    
    # Convert to dB, which is a log scale.  Use peak power as reference.
    logMelSpectrogram = librosa.power_to_db(melSpectrum, ref=np.max)
    
    return logMelSpectrogram




def plotLogMelSpectrogram(plotTitle, sampleRate, logMelSpectrum, figWidth=14, figHeight=4):
    fig = plt.figure(figsize=(figWidth, figHeight))
    librosa.display.specshow(logMelSpectrum, sr=sampleRate, x_axis='time', y_axis='mel')
    plt.title('Mel log-frequency power spectrogram: ' + plotTitle)
    plt.colorbar(pad=0.01, format='%+02.0f dB')
    plt.tight_layout()  
    plt.show()  # force display while in for loop
    return None




def computeMFCC(samples, sampleRate, nFFT=512, hopLength=256, nMFCC=40):
    mfcc = librosa.feature.mfcc(y=samples, sr=sampleRate, 
                                n_fft=nFFT, hop_length=hopLength, n_mfcc=nMFCC)
    
    # Let's add on the first and second deltas  (what is this really doing?)
    #mfcc = librosa.feature.delta(mfcc, order=2)
    return mfcc




def plotMFCC(plotTitle, sampleRate, mfcc, figWidth=14, figHeight=4):
    fig = plt.figure(figsize=(figWidth, figHeight))
    librosa.display.specshow(mfcc, sr=sampleRate, x_axis='time', y_axis='mel')
    plt.colorbar(pad=0.01)
    plt.title("Mel-frequency cepstral coefficients (MFCC): " + plotTitle)
    plt.tight_layout()
    plt.show()  # force display while in for loop
    return None




def showWavefile(filename):
    sampleRate, samples = wavfile.read(filename)  
    plotRawWave(filename, sampleRate, samples)
    
    freqs, times, logSpectrogram = computeLogSpectrogram(samples, sampleRate)
    plotLogSpectrogram(filename, freqs, times, logSpectrogram)
    
    logMelSpectrogram = computeLogMelSpectrogram(samples, sampleRate)
    plotLogMelSpectrogram(filename, sampleRate, logMelSpectrogram)
    
    mfcc = computeMFCC(samples, sampleRate)
    #print(mfcc.shape)
    plotMFCC(filename, sampleRate, mfcc)
    
    return sampleRate, samples, logSpectrogram, logMelSpectrogram, mfcc




labels = ['filename', 'comments']
waves  = [
          ('seven/712e4d58_nohash_2.wav',  # 10
           'Noise.'),
    
          ('seven/099d52ad_nohash_4.wav',  # 29
           'Noise.'),

          ('seven/ced835d3_nohash_3.wav',  # 132
           'Unintelligible, noise.'),

          ('seven/7846fd85_nohash_0.wav',  # 265
           'Unintelligible, noise.'),

          ('seven/aff582a1_nohash_0.wav',  # 591
           'clicks with noise.' ),
 
          ('seven/3c165869_nohash_0.wav',  # 1049
           '"seven"'),  
    
          ('seven/e82914c0_nohash_0.wav',  # 2084
           '"seven"'),

          ('seven/fb7cfe0e_nohash_0.wav',  # 4114
           '"seven"'),
                                                                                                    
          ('seven/9ff1b8b6_nohash_1.wav',  # 8193
           'Muffled "seven"'),
    
          ('seven/28ed6bc9_nohash_4.wav',  # 16440
           '"seven"'),

           ('seven/471a0925_nohash_3.wav',  # 32800
           'noisy "seven"'),
                                                                                  
          ('seven/facd97c0_nohash_0.wav',  # 65535
           'LOUD "seven"')
         ]                  

wavedf = pd.DataFrame.from_records(waves, columns=labels)
wavedf




for i in range(wavedf.shape[0]):
    
    filenameShort = wavedf.loc[i, 'filename']
    word, wavename = filenameShort.split("/")
    
    filename = trainAudioPath + filenameShort
    
    sampleRate, samples = wavfile.read(filename)  
    caption = '"' + word  +     '" [amplitude from ' + str(min(samples)) +  ' to ' + str(max(samples)) +     ', range = ' + str(int(max(samples)) - int(min(samples))) + ']'
    
    printMarkdown("# " + caption)
    print(wavedf.loc[i, 'comments'])
    
    ipd.display( ipd.Audio(filename) )
    sampleRate, samples, logSpectrogram, logMelSpectrogram, mfcc = showWavefile(filename)






