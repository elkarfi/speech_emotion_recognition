#!/usr/bin/env python
# coding: utf-8

# ## Importation des bibliotheques necessaires

import librosa
import librosa.display
import tkinter as tk
from matplotlib import cm
import scipy.io.wavfile as wav
import scipy.signal as signal
import seaborn as sns
import IPython.display as ipd
import matplotlib.pyplot as plt
import os, glob
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# ## Extraction des caractéristiques  d'un audio


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft=np.abs(librosa.stft(X))
    result=np.array([])
    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    result=np.hstack((result, mfccs))
    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    result=np.hstack((result, chroma))
    mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    result=np.hstack((result, mel))
    return result


# ##  Les 8 émotions contenus dans RAVDESS et les 3 émotions à observer


emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions=['happy', 'sad','angry','neutral']


# ## Charger les données et extraire les infromations


def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("D:\\Project\\ravdess data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=42)


# ## Répartir les données (Apprentissage et Test)


x_train,x_test,y_train,y_test=load_data(test_size=0.2)



plt.title('Nombre des Emotions', size=16)
sns.countplot(y_train)
plt.ylabel('Nombre', size=12)
plt.xlabel('Emotion', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()


# ## La taille de la base d'apprentissage et de la base de test



print((x_train.shape[0], x_test.shape[0]))


# ## Accuracy vs le nombre des neurons


acc = []
acc_tr = []
timelog = []
for l in [10,20,50,100,200,500,1000]:
    t = time()
    #Initialisation des paramètres du Multi Layer Perceptron Classifier
    model = MLPClassifier(alpha=0.001, batch_size=256,  hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    #Entrainer le modèle
    model.fit(x_train,y_train)
    endt = time() - t
    #Prédire les émotions    
    a_tr = accuracy_score(y_train, model.predict(x_train)) # Train Accuracy
    a = accuracy_score(y_test, model.predict(x_test)) # Test Accuracy
    a_tr
    a
    acc_tr.append(a_tr)
    acc.append(a)
    timelog.append(endt)




l = [10,20,50,100,200,500,1000]
N = len(l)
l2 = np.arange(N)
plt.subplots(figsize=(10, 5))
plt.plot(l2, acc, label="Test Accuracy")
plt.plot(l2, acc_tr, label="Train Accuracy")
plt.xticks(l2,l)
plt.grid(True)
plt.xlabel("Neurons")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# # Exemple de prédiction

# ## Initialisation des paramètres du Multi Layer Perceptron Classifier


model = MLPClassifier(alpha=0.001, batch_size=256,  hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)


# ## Entrainer le modèle



model.fit(x_train,y_train)


# ## Prédire les émotions


y_pred=model.predict(x_test)


# ## Évaluer la performance du modèle


accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)


print("Accuracy: {:.2f}%".format(accuracy*100))
print(y_pred)


file="D://Project//ravdess data//Actor_01//03-01-05-02-01-01-01.wav" 
feature=extract_feature(file)



ipd.Audio(file)


x, sr = librosa.load(file)


print (x[40000])
y=len(x)
print(sr)
print(y)


plt.figure(figsize=(8, 4))
librosa.display.waveplot(x, sr=sr)
plt.ylabel("Amplitude")
plt.xlabel("Temps(s)")


spectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128,fmax=8000) 
spectrogram = librosa.power_to_db(spectrogram)
librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time');
plt.ylabel("frequence")
plt.xlabel("Temps")
plt.colorbar()



mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=4)
librosa.display.specshow(mfccs, x_axis='time')
plt.ylabel("Amplitude")
plt.xlabel("Temps")
plt.colorbar()
plt.tight_layout()
plt.title('mfcc')
plt.show



sample_rate, samples = wav.read(file)
f, t, Zxx = signal.stft(samples, fs=sample_rate)
plt.pcolormesh(t, f, np.abs(Zxx))


y_pred=model.predict([feature])
print(y_pred)


master = tk.Tk()
master.title('Speech Emotion Recognition')
master.geometry("500x300")
w = tk.Label(master, text="le fichier audio choisi appartient à la classe :") 
w.pack(fill=tk.BOTH, expand=1, padx=100, pady=50)
z = tk.Label(master, text="Angry", font=('Century 20 bold')) 
z.pack(fill=tk.BOTH, expand=1, padx=100, pady=50)
master.mainloop()