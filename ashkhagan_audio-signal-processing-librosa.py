#!/usr/bin/env python
# coding: utf-8



#Sound Envelope
#- Attack-decay-sustain-release-model
from IPython.display import IFrame
IFrame('https://tonejs.github.io/examples/envelope.html', width=700, height=350)




pwd()




cd ../input/birdsong-recognition/train_audio/aldfly




ls




get_ipython().system('pip install ffmpeg')




#loading an audio file into an audio array
import librosa
x, sr = librosa.load('XC134874.mp3')




#To display the length of the audio array and samplerate
print(x.shape)
print(sr)




get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import librosa.display




#plotting the audio array with librosa.display.wavplot
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)




#displaying the spectrogram using librosa.display.specshow
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
import IPython.display as ipd
ipd.Audio('XC134874.mp3') # load a local WAV file




get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
import numpy, scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
import librosa, librosa.display




import warnings
warnings.filterwarnings("ignore")




signals = [
    librosa.load(p)[0] for p in Path().glob('XC*.mp3')
]




len(signals)




plt.figure(figsize=(100, 500))
for i, x in enumerate(signals):
    plt.subplot(25, 4, i+1)
    librosa.display.waveplot(x[:10000])
    plt.ylim(-1, 1)




#let's now construct a feature vector >feature vector is simply a collection of features.
def extract_features(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0, 0],
        librosa.feature.spectral_centroid(signal)[0, 0],
    ]




features = numpy.array([extract_features(x) for x in signals])




plt.figure(figsize=(14, 5))
plt.hist(features[:,0], color='b', range=(0, 0.2), alpha=0.5, bins=20)
plt.legend(('Aldfly signals'))
plt.xlabel('Zero Crossing Rate')
plt.ylabel('Count')




plt.figure(figsize=(14, 5))
plt.hist(features[:,1], color='b', range=(0, 4000), bins=30, alpha=0.6)
plt.legend(('Aldfly Signals'))
plt.xlabel('Spectral Centroid (frequency bin)')
plt.ylabel('Count')




#this part does the feature scaling so that those fetures can worked on with each other
#so like what we do here is bringing them all in range of -1,1 as neccessary




feature_table = numpy.vstack((features))
print(feature_table.shape)




scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
training_features = scaler.fit_transform(feature_table)
print(training_features.min(axis=0))
print(training_features.max(axis=0))




plt.scatter(training_features[:50,0], training_features[:50,1], c='b')
plt.scatter(training_features[50:,0], training_features[50:,1], c='r')
plt.xlabel('Zero Crossing Rate')
plt.ylabel('Spectral Centroid')




#Segmentation 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display
plt.rcParams['figure.figsize'] = (11, 5)




T = 3.0      # duration in seconds
sr = 22050   # sampling rate in Hertz
amplitude = numpy.logspace(-3, 0, int(T*sr), endpoint=False, base=10.0) # time-varying amplitude
print amplitude.min(), amplitude.max()
# starts at 110 Hz, ends at 880 Hz
#create the signal
t = numpy.linspace(0, T, int(T*sr), endpoint=False)
x = amplitude*numpy.sin(2*numpy.pi*440*t)

ipd.Audio(x, rate=sr)
#Plot the signal:
librosa.display.waveplot(x, sr=sr)




#Segmentation Using Python List Comprehensions
#In Python, you can use a standard list comprehension to perform segmentation of a signal and compute RMSE at the same time.
#Initialize segmentation parameters:

frame_length = 1024
hop_length = 512
#Define a helper function:

def rmse(x):
    return numpy.sqrt(numpy.mean(x**2))

#Using a list comprehension, plot the RMSE for each frame on a log-y axis:

plt.semilogy([rmse(x[i:i+frame_length])
              for i in range(0, len(x), hop_length)])

#librosa.util.frame
#Given a signal, librosa.util.frame will produce a list of uniformly sized frames:

frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length)
plt.semilogy([rmse(frame) for frame in frames.T])

#That being said, in librosa, manual segmentation of a signal is often unnecessary, because the feature extraction methods themselves do segmentation for you.




#Zero Crossing Rate
#The zero crossing rate indicates the number of times that a signal crosses the horizontal axis.
#Let's load a signal:

x, sr = librosa.load('XC134874.mp3')
#Listen to the signal:
ipd.Audio(x, rate=sr)

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)




#Let's zoom in:

n0 = 6500
n1 = 7500
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])




#I count five zero crossings. Let's compute the zero crossings using librosa.

zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
zero_crossings.shape
#That computed a binary mask where True indicates the presence of a zero crossing. To find the total number of zero crossings, use sum:
print(sum(zero_crossings))




#To find the zero-crossing rate over time, use zero_crossing_rate:
zcrs = librosa.feature.zero_crossing_rate(x)
print(zcrs.shape)




plt.figure(figsize=(14, 5))
plt.plot(zcrs[0])




#Note how the high zero-crossing rate corresponds to the presence of the snare drum.

#The reason for the high rate near the beginning is because the silence oscillates quietly around zero:

plt.figure(figsize=(14, 5))
plt.plot(x[:1000])
plt.ylim(-0.0001, 0.0001)




#A simple hack around this is to add a small constant before computing the zero crossing rate:

zcrs = librosa.feature.zero_crossing_rate(x + 0.0001)
plt.figure(figsize=(14, 5))
plt.plot(zcrs[0])




import seaborn
import numpy, scipy, matplotlib.pyplot as plt, librosa, IPython.display as ipd




import urllib
filename = 'XC134874.mp3'
x, sr = librosa.load(filename)




print x.shape
print sr




ipd.Audio(x, rate=sr)




X = scipy.fft(x)
X_mag = numpy.absolute(X)
f = numpy.linspace(0, sr, len(X_mag)) # frequency variable




#Plot the spectrum:

plt.figure(figsize=(13, 5))
plt.plot(f, X_mag) # magnitude spectrum
plt.xlabel('Frequency (Hz)')




#Zoom in:
plt.figure(figsize=(13, 5))
plt.plot(f[:5000], X_mag[:5000])
plt.xlabel('Frequency (Hz)')




#loadingfile
x, sr = librosa.load('XC134874.mp3')
ipd.Audio(x, rate=sr)




#librosa.stft computes a STFT. 
#We provide it a frame size, i.e. the size of the FFT, and a hop length, i.e. the frame increment:
hop_length = 512
n_fft = 2048
X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)




#To convert the hop length and frame size to units of seconds:

float(hop_length)/sr # units of seconds




float(n_fft)/sr  # units of seconds




#For real-valued signals, the Fourier transform is symmetric about the midpoint. Therefore, librosa.stft only retains one half of the output:
X.shape
#This STFT has 1025 frequency bins and 9813 frames in time.




S = librosa.amplitude_to_db(abs(X))




plt.figure(figsize=(15, 5))
librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')




hop_length = 256
S = librosa.feature.melspectrogram(x, sr=sr, n_fft=4096, hop_length=hop_length)




logS = librosa.power_to_db(abs(S))




plt.figure(figsize=(15, 5))
librosa.display.specshow(logS, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')




fmin = librosa.midi_to_hz(36)
C = librosa.cqt(x, sr=sr, fmin=fmin, n_bins=72)
logC = librosa.amplitude_to_db(abs(C))




plt.figure(figsize=(15, 5))
librosa.display.specshow(logC, sr=sr, x_axis='time', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')




ipd.Audio(x, rate=sr)




fmin = librosa.midi_to_hz(36)
hop_length = 512
C = librosa.cqt(x, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)




# Display:
logC = librosa.amplitude_to_db(numpy.abs(C))
plt.figure(figsize=(15, 5))
librosa.display.specshow(logC, sr=sr, x_axis='time', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
#Note how each frequency bin corresponds to one MIDI pitch number.




chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')




chromagram = librosa.feature.chroma_cqt(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')




chromagram = librosa.feature.chroma_cens(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')




T = 4.0      # duration in seconds
sr = 22050   # sampling rate in Hertz
t = numpy.linspace(0, T, int(T*sr), endpoint=False)




#Create a signal whose amplitude grows linearly:
amplitude = numpy.linspace(0, 1, int(T*sr), endpoint=False) # time-varying amplitude
x = amplitude*numpy.sin(2*numpy.pi*440*t)




librosa.display.waveplot(x, sr=sr)




# Now consider a signal whose amplitude grows exponentially, i.e. the logarithm of the amplitude is linear:

amplitude = numpy.logspace(-2, 0, int(T*sr), endpoint=False, base=10.0)
x = amplitude*numpy.sin(2*numpy.pi*440*t)




librosa.display.waveplot(x, sr=sr)




x, sr = librosa.load('XC134874.mp3', duration=25)
ipd.Audio(x, rate=sr)




X = librosa.stft(x)
X.shape




#Raw amplitude:
Xmag = abs(X)
librosa.display.specshow(Xmag, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()




Xdb = librosa.amplitude_to_db(Xmag)
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()




Xmag = numpy.log10(1+10*abs(X))
librosa.display.specshow(Xmag, sr=sr, x_axis='time', y_axis='log', cmap="gray_r")
plt.colorbar()




freqs = librosa.core.fft_frequencies(sr=sr)
Xmag = librosa.perceptual_weighting(abs(X)**2, freqs)
librosa.display.specshow(Xmag, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()




get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd, sklearn
import librosa, librosa.display
plt.rcParams['figure.figsize'] = (14, 5)




x, sr = librosa.load('XC134874.mp3')
ipd.Audio(x, rate=sr)




spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape




#Compute the time variable for visualization:

frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)




def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)




librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r') # normalize for visualization purposes




spectral_centroids = librosa.feature.spectral_centroid(x+0.01, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r') # normalize for visualization purposes




spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
plt.legend(('p = 2', 'p = 3', 'p = 4'))




spectral_contrast = librosa.feature.spectral_contrast(x, sr=sr)
spectral_contrast.shape




plt.imshow(normalize(spectral_contrast, axis=1), aspect='auto', origin='lower', cmap='coolwarm')




spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')




x, sr = librosa.load('XC134874.mp3')
ipd.Audio(x, rate=sr)




plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr)




# Because the autocorrelation produces a symmetric signal, we only care about the "right half".
r = numpy.correlate(x, x, mode='full')[len(x)-1:]
print(x.shape, r.shape)




#Plot the autocorrelation:
plt.figure(figsize=(14, 5))
plt.plot(r[:10000])
plt.xlabel('Lag (samples)')
plt.xlim(0, 10000)




r = librosa.autocorrelate(x, max_size=10000)
print(r.shape)




plt.figure(figsize=(14, 5))
plt.plot(r)
plt.xlabel('Lag (samples)')
plt.xlim(0, 10000)




x, sr = librosa.load('XC134874.mp3')
ipd.Audio(x, rate=sr)




#Compute and plot the autocorrelation:
r = librosa.autocorrelate(x, max_size=5000)
plt.figure(figsize=(14, 5))
plt.plot(r[:200])




midi_hi = 120.0
midi_lo = 12.0
f_hi = librosa.midi_to_hz(midi_hi)
f_lo = librosa.midi_to_hz(midi_lo)
t_lo = sr/f_hi
t_hi = sr/f_lo




print(f_lo, f_hi)
print(t_lo, t_hi)




r[:int(t_lo)] = 0
r[int(t_hi):] = 0
plt.figure(figsize=(14, 5))
plt.plot(r[:1400])




get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn
import numpy, scipy, IPython.display as ipd, matplotlib.pyplot as plt
import librosa, librosa.display
plt.rcParams['figure.figsize'] = (14, 5)




filename = 'XC134874.mp3'
x, sr = librosa.load(filename)




#Play the audio file.
ipd.Audio(x, rate=sr)




#Display the CQT of the signal.
bins_per_octave = 36
cqt = librosa.cqt(x, sr=sr, n_bins=300, bins_per_octave=bins_per_octave)
log_cqt = librosa.logamplitude(cqt)
cqt.shape




librosa.display.specshow(log_cqt, sr=sr, x_axis='time', y_axis='cqt_note', 
                         bins_per_octave=bins_per_octave)




hop_length = 100
onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=hop_length)
plt.plot(onset_env)
plt.xlim(0, len(onset_env))




onset_samples = librosa.onset.onset_detect(x,
                                           sr=sr, units='samples', 
                                           hop_length=hop_length, 
                                           backtrack=False,
                                           pre_max=20,
                                           post_max=20,
                                           pre_avg=100,
                                           post_avg=100,
                                           delta=0.2,
                                           wait=0)




onset_samples




#Let's pad the onsets with the beginning and end of the signal.
onset_boundaries = numpy.concatenate([[0], onset_samples, [len(x)]])
print(onset_boundaries)




#Convert the onsets to units of seconds:
onset_times = librosa.samples_to_time(onset_boundaries, sr=sr)
onset_times




#Display the results of the onset detection:
librosa.display.waveplot(x, sr=sr)
plt.vlines(onset_times, -1, 1, color='r')




def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):
    
    # Compute autocorrelation of input segment.
    r = librosa.autocorrelate(segment)
    
    # Define lower and upper limits for the autocorrelation argmax.
    i_min = sr/fmax
    i_max = sr/fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0
    
    # Find the location of the maximum autocorrelation.
    i = r.argmax()
    f0 = float(sr)/i
    return f0




def generate_sine(f0, sr, n_duration):
    n = numpy.arange(n_duration)
    return 0.2*numpy.sin(2*numpy.pi*f0*n/float(sr))




def estimate_pitch_and_generate_sine(x, onset_samples, i, sr):
    n0 = onset_samples[i]
    n1 = onset_samples[i+1]
    f0 = estimate_pitch(x[n0:n1], sr)
    return generate_sine(f0, sr, n1-n0)




y = numpy.concatenate([
    estimate_pitch_and_generate_sine(x, onset_boundaries, i, sr=sr)
    for i in range(len(onset_boundaries)-1)
])




#Play the synthesized transcription.
ipd.Audio(y, rate=sr)




#Plot the CQT of the synthesized transcription.
cqt = librosa.cqt(y, sr=sr)
librosa.display.specshow(abs(cqt), sr=sr, x_axis='time', y_axis='cqt_note')




x, sr = librosa.load('XC134874.mp3')
print(x.shape, sr)




#Plot the signal:

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr)
#Listen:
ipd.Audio(x, rate=sr)




hop_length = 512
frame_length = 1024
rmse = librosa.feature.rmse(x, frame_length=frame_length, hop_length=hop_length).flatten()
rmse_diff = numpy.zeros_like(rmse)
rmse_diff[1:] = numpy.diff(rmse)
print(rmse.shape)
print(rmse_diff.shape)




energy_novelty = numpy.max([numpy.zeros_like(rmse_diff), rmse_diff], axis=0)




#Plot all three functions together:

frames = numpy.arange(len(rmse))
t = librosa.frames_to_time(frames, sr=sr)
plt.figure(figsize=(15, 6))
plt.plot(t, rmse, 'b--', t, rmse_diff, 'g--^', t, energy_novelty, 'r-')
plt.xlim(0, t.max())
plt.xlabel('Time (sec)')
plt.legend(('RMSE', 'delta RMSE', 'energy novelty')) 




log_rmse = numpy.log1p(10*rmse)
log_rmse_diff = numpy.zeros_like(log_rmse)
log_rmse_diff[1:] = numpy.diff(log_rmse)
log_energy_novelty = numpy.max([numpy.zeros_like(log_rmse_diff), log_rmse_diff], axis=0)
plt.figure(figsize=(15, 6))
plt.plot(t, log_rmse, 'b--', t, log_rmse_diff, 'g--^', t, log_energy_novelty, 'r-')
plt.xlim(0, t.max())
plt.xlabel('Time (sec)')
plt.legend(('log RMSE', 'delta log RMSE', 'log energy novelty')) 




sr = 22050
def generate_tone(midi):
    T = 0.5
    t = numpy.linspace(0, T, int(T*sr), endpoint=False)
    f = librosa.midi_to_hz(midi)
    return numpy.sin(2*numpy.pi*f*t)
x = numpy.concatenate([generate_tone(midi) for midi in [48, 52, 55, 60, 64, 67, 72, 76, 79, 84]])




# Listen:
ipd.Audio(x, rate=sr)




#The energy novelty function remains roughly constant:
hop_length = 512
frame_length = 1024
rmse = librosa.feature.rmse(x, frame_length=frame_length, hop_length=hop_length).flatten()
rmse_diff = numpy.zeros_like(rmse)
rmse_diff[1:] = numpy.diff(rmse)
energy_novelty = numpy.max([numpy.zeros_like(rmse_diff), rmse_diff], axis=0)
frames = numpy.arange(len(rmse))
t = librosa.frames_to_time(frames, sr=sr)
plt.figure(figsize=(15, 4))
plt.plot(t, rmse, 'b--', t, rmse_diff, 'g--^', t, energy_novelty, 'r-')
plt.xlim(0, t.max())
plt.xlabel('Time (sec)')
plt.legend(('RMSE', 'delta RMSE', 'energy novelty')) 




spectral_novelty = librosa.onset.onset_strength(x, sr=sr)
frames = numpy.arange(len(spectral_novelty))
t = librosa.frames_to_time(frames, sr=sr)
plt.figure(figsize=(15, 4))
plt.plot(t, spectral_novelty, 'r-')
plt.xlim(0, t.max())
plt.xlabel('Time (sec)')
plt.legend(('Spectral Novelty',))




x, sr = librosa.load('XC134874.mp3')
print(x.shape, sr)




#Listen to the audio file:

ipd.Audio(x, rate=sr)




#Plot the signal:
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)




#Compute an onset envelope:

hop_length = 256
onset_envelope = librosa.onset.onset_strength(x, sr=sr, hop_length=hop_length)
onset_envelope.shape




#Generate a time variable:

N = len(x)
T = N/float(sr)
t = numpy.linspace(0, T, len(onset_envelope))
#Plot the onset envelope:

plt.figure(figsize=(14, 5))
plt.plot(t, onset_envelope)
plt.xlabel('Time (sec)')
plt.xlim(xmin=0)
plt.ylim(0)




onset_frames = librosa.util.peak_pick(onset_envelope, 7, 7, 7, 7, 0.5, 5)
onset_frames




#Plot the onset envelope along with the detected peaks:

plt.figure(figsize=(14, 5))
plt.plot(t, onset_envelope)
plt.grid(False)
plt.vlines(t[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.7)
plt.xlabel('Time (sec)')
plt.xlim(0, T)
plt.ylim(0)




#superimpose a click track upon the original:

clicks = librosa.clicks(frames=onset_frames, sr=22050, hop_length=hop_length, length=N)
ipd.Audio(x+clicks, rate=sr)




x, sr = librosa.load('XC134874.mp3')




onset_frames = librosa.onset.onset_detect(x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
print(onset_frames) # frame numbers of estimated onsets




#Convert onsets to units of seconds:

onset_times = librosa.frames_to_time(onset_frames)
print(onset_times)




#Plot the onsets on top of a spectrogram of the audio:

S = librosa.stft(x)
logS = librosa.amplitude_to_db(abs(S))




clicks = librosa.clicks(frames=onset_frames, sr=sr, length=len(x))




ipd.Audio(x + clicks, rate=sr)




ipd.Audio(numpy.vstack([x, clicks]), rate=sr)




#You can also change the click to a custom audio file instead:

cowbell, _ = librosa.load('XC134874.mp3')




#More cowbell?

clicks = librosa.clicks(frames=onset_frames, sr=sr, length=len(x), click=cowbell)
ipd.Audio(x + clicks, rate=sr)




x, sr = librosa.load('XC134874.mp3')
print(x.shape, sr)




#Listen:

ipd.Audio(x, rate=sr)




#Compute the frame indices for estimated onsets in a signal:
hop_length = 512
onset_frames = librosa.onset.onset_detect(x, sr=sr, hop_length=hop_length)
print(onset_frames) 
# frame numbers of estimated onsets




#Convert onsets to units of seconds:
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
print(onset_times)




#Convert onsets to units of samples:
onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
print(onset_samples)




#Plot the onsets on top of a spectrogram of the audio:
S = librosa.stft(x)
logS = librosa.logamplitude(S)
librosa.display.specshow(logS, sr=sr, x_axis='time', y_axis='log')
plt.vlines(onset_times, 0, 10000, color='k')




def concatenate_segments(x, onset_samples, pad_duration=0.500):
    """Concatenate segments into one signal."""
    silence = numpy.zeros(int(pad_duration*sr)) # silence
    frame_sz = min(numpy.diff(onset_samples))   # every segment has uniform frame size
    return numpy.concatenate([
        numpy.concatenate([x[i:i+frame_sz], silence]) # pad segment with silence
        for i in onset_samples
    ])




#Concatenate the segments:
concatenated_signal = concatenate_segments(x, onset_samples, 0.500)
#Listen to the concatenated signal
ipd.Audio(concatenated_signal, rate=sr)




onset_frames = librosa.onset.onset_detect(x, sr=sr, hop_length=hop_length, backtrack=True)
#Convert onsets to units of seconds:

onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
#Convert onsets to units of samples:

onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
#Plot the onsets on top of a spectrogram of the audio:

S = librosa.stft(x)
logS = librosa.logamplitude(S)
librosa.display.specshow(logS, sr=sr, x_axis='time', y_axis='log')
plt.vlines(onset_times, 0, 10000, color='k')




concatenated_signal = concatenate_segments(x, onset_samples, 0.500)
#Listen to the concatenated signal:

ipd.Audio(concatenated_signal, rate=sr)
#While listening, notice now the segments are perfectly segmented.




x, sr = librosa.load('XC134874.mp3')
ipd.Audio(x, rate=sr)




hop_length = 200 # samples per frame
onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=hop_length, n_fft=2048)




#Plot the onset envelope:

frames = range(len(onset_env))
t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
plt.plot(t, onset_env)
plt.xlim(0, t.max())
plt.ylim(0)
plt.xlabel('Time (sec)')
plt.title('Novelty Function')




#Compute the short-time Fourier transform (STFT) of the novelty function. Since the novelty function is computed in frame increments, the hop length of this STFT should be pretty small:

S = librosa.stft(onset_env, hop_length=1, n_fft=512)
fourier_tempogram = numpy.absolute(S)
#Plot the Fourier tempogram:

librosa.display.specshow(fourier_tempogram, sr=sr, hop_length=hop_length, x_axis='time')




n0 = 100
n1 = 500
plt.plot(t[n0:n1], onset_env[n0:n1])
plt.xlim(t[n0], t[n1])
plt.xlabel('Time (sec)')
plt.title('Novelty Function')




#Plot the autocorrelation of this segment:

tmp = numpy.log1p(onset_env[n0:n1])
r = librosa.autocorrelate(tmp)
plt.plot(t[:n1-n0], r)
plt.xlim(t[0], t[n1-n0])
plt.xlabel('Lag (sec)')
plt.ylim(0)




#Wherever the autocorrelation is high is a good candidate of the beat period.

plt.plot(60/t[:n1-n0], r)
plt.xlim(20, 200)
plt.xlabel('Tempo (BPM)')
plt.ylim(0)




#We will apply this principle of autocorrelation to estimate the tempo at every segment in the novelty function.

#librosa.feature.tempogram implements an autocorrelation tempogram, a short-time autocorrelation of the (spectral) novelty function.

#For more information:

#Grosche, Peter, Meinard Müller, and Frank Kurth. “Cyclic tempogram - A mid-level tempo representation for music signals.” ICASSP, 2010.
#Compute a tempogram:

tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=400)
librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo')




tempo = librosa.beat.tempo(x, sr=sr)
print(tempo)




#Visualize the tempo estimate on top of the input signal:

T = len(x)/float(sr)
seconds_per_beat = 60.0/tempo[0]
beat_times = numpy.arange(0, T, seconds_per_beat)
librosa.display.waveplot(x)
plt.vlines(beat_times, -1, 1, color='r')




#Listen to the input signal with a click track using the tempo estimate:

clicks = librosa.clicks(beat_times, sr, length=len(x))
ipd.Audio(x + clicks, rate=sr)




x, sr = librosa.load('XC134874.mp3')
ipd.Audio(x, rate=sr)




tempo, beat_times = librosa.beat.beat_track(x, sr=sr, start_bpm=60, units='time')
print(tempo)
print(beat_times)




#Plot the beat locations over the waveform:

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, alpha=0.6)
plt.vlines(beat_times, -1, 1, color='r')
plt.ylim(-1, 1)




#Plot a histogram of the intervals between adjacent beats:

beat_times_diff = numpy.diff(beat_times)
plt.figure(figsize=(14, 5))
plt.hist(beat_times_diff, bins=50, range=(0,4))
plt.xlabel('Beat Length (seconds)')
plt.ylabel('Count')




#Visually, it's difficult to tell how correct the estimated beats are. Let's listen to a click track:

clicks = librosa.clicks(beat_times, sr=sr, length=len(x))
ipd.Audio(x + clicks, rate=sr)




def f(start_bpm, tightness_exp):
    return librosa.beat.beat_track(x, sr=sr, start_bpm=start_bpm, tightness=10**tightness_exp, units='time')
interact(f, start_bpm=60, tightness_exp=2)

