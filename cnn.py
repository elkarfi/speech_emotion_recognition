#!/usr/bin/env python
# coding: utf-8

# ## Importation des bibliotheques necessaires

import librosa
import librosa.display
import tkinter as tk
import seaborn as sns
import IPython.display as ipd
import matplotlib.pyplot as plt
import os, glob
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
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

observed_emotions=['calm','happy', 'sad','angry']


# ## Charger les données et extraire les infromations


def load_data(test_size):
    x,y=[],[]
    for file in glob.glob("D:\\Project\\ravdess data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file)
        x.append(feature)
        if emotion =='calm':
            y.append(0)
        if emotion =='sad':
            y.append(1)
        if emotion =='angry':
            y.append(2)
        if emotion =='happy':
            y.append(3)
    return train_test_split(np.asarray(x), np.asarray(y), test_size=test_size, random_state=42,shuffle=True)


# ## Répartir les données (Apprentissage et Test)

x_train,x_test,y_train,y_test=load_data(test_size=0.33)


# ## Distibution des Emotions

plt.title('Distibution des Emotions', size=16)
sns.countplot(y_train)
plt.ylabel('Nombre', size=12)
plt.xlabel('Emotion', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()


# ## La taille de la base d'apprentissage et de la base de test

x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# # CNN


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, AveragePooling1D,Activation
from tensorflow.keras.regularizers import l2



model = Sequential()
model.add(Conv1D(64, kernel_size=(10), activation='relu', input_shape=(x_train.shape[1],1)))
model.add(Conv1D(128, kernel_size=(10),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Dropout(0.4))
model.add(Conv1D(128, kernel_size=(10),activation='relu'))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4, activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,metrics=['accuracy'])



history=model.fit(x_train, y_train, batch_size=16, epochs=200, validation_data=(x_test, y_test))



score= model.evaluate(x_test, y_test, verbose = 1)
accuracy = 100*score[1]


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


predictions = model.predict_classes(x_test)



predictions


y_test



file="D://Project//ravdess data//Actor_01//03-01-05-02-01-01-01.wav" 
feature=extract_feature(file)
#feature=np.asarray(feature)
#feature = np.expand_dims(feature, axis=0)
#feature


feature.shape


feature = np.expand_dims(feature, axis=(0,2))


feature.shape


ypred = model.predict_classes(feature)
print(ypred)




master = tk.Tk()
master.title('Speech Emotion Recognition')
master.geometry("500x300")
w = tk.Label(master, text="le fichier audio choisi appartient à la classe :") 
w.pack(fill=tk.BOTH, expand=1, padx=100, pady=50)
z = tk.Label(master, text="Angry", font=('Century 20 bold')) 
z.pack(fill=tk.BOTH, expand=1, padx=100, pady=50)
master.mainloop()






