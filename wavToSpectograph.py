import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import pathlib
import numpy as np

#amount of samples to skip after fft
#this is set to librosa default value for now
hopLength = 512
#number of samples per fft
#this is set to librosa default value for now
n_fft = 2048

def audioToSpectograph(audioFilePath):
    #get list of audio clips from file path
    audioClips = os.listdir(audioFilePath)
    #iterate through audio clips
    for i in audioClips:
        #get signal and audio sampling rate
        signal, samplingRate = librosa.load(audioFilePath+audioClips[i])

        #get mel spectogram signal
        melSpecSignal = librosa.feature.melspectrogram(signal, samplingRate, hopLength, n_fft)
        #get absolute values for all values in melSpecSignal
        spectogram = np.abs(melSpecSignal)
        #apply log transformation
        logSpectogram = librosa.power_to_db(spectogram, np.max)

        #set figsize
        plt.figure(figsize=(10,8))
        #create mel spectogram
        librosa.display.specshow(logSpectogram, samplingRate, x_axis='time', y_axis='mel', hop_length=hopLength, n_fft=n_fft)

def main():
    pathlib.Path("rawVoiceData")

    


