###
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
###



import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.datasets import cifar10
from sklearn.model_selection import cross_val_score
from util import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam   
import time
from keras.utils import to_categorical
from random import shuffle

def create_model_task_3(num_classes,input_shape):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_shape,)))
    
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    return model

def create_model_task_4(num_classes,input_shape):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    return model

def preprocessing(X):
    X_prec = []

    for item in X:
        gray_image = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)

        X_prec.append(gray_image)

    X_prec = np.array(X_prec)
    
    X_prec = X_prec.reshape((-1, 32 * 32))

    return X_prec

num_classes=10
channels=3
img_size=32
input_shape=img_size*img_size*channels

def third_task(x_train, x_test, y_train, y_test):
    """
		
		Move on to Neural Networks, using one hidden layer. You should 
		numerically check your gradient calculations.

	"""


    print('x_train shape:', x_train.shape)
    print(x_train.shape, 'train samples')
    print(x_test.shape, 'test samples')

    batch_size=128
    epochs=50

    #X_ = np.concatenate((x_train, x_test))
    #y_ = np.concatenate((y_train, y_test))
    y_ = [item[0] for item in y_train]
    y_ = np.array(y_)


    #X_ = preprocessing(x_train)
    X_ = x_train
    # init the variables
    accuracies,precisions,recalls, f1s=[],[],[],[]
    
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    number = 0
    start = time.time()
    for train_index, val_index in skf.split(X_, y_):

        X_train, X_test = X_[train_index], X_[val_index]
        y_train_, y_test_ = y_[train_index], y_[val_index]
        y_train_ = to_categorical(y_train_, num_classes)
        y_test_ = to_categorical(y_test_, num_classes)

        
        model = create_model_task_3(num_classes,input_shape)
        model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001),
                  metrics=['accuracy'])

        history = model.fit(X_train, y_train_,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test_))
        score = model.evaluate(X_test, y_test_, verbose=0)


        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        number += 1

    print("it took", time.time() - start, "seconds.")

def fourth_task(x_train, x_test, y_train, y_test):
    """
        
        Extend your Neural Network to two hidden layers. 
        Try diferent activation functions. Does the 
        performance improve...


    """


    print('x_train shape:', x_train.shape)
    print(x_train.shape, 'train samples')
    print(x_test.shape, 'test samples')

    batch_size=128
    epochs=50

    y_ = [item[0] for item in y_train]
    y_ = np.array(y_)

    #X_ = preprocessing(x_train)
    X_ = x_train
    # init the variables
    accuracies,precisions,recalls, f1s=[],[],[],[]
    
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    number = 0
    start = time.time()
    for train_index, val_index in skf.split(X_, y_):

        X_train, X_test = X_[train_index], X_[val_index]
        y_train_, y_test_ = y_[train_index], y_[val_index]
        y_train_ = to_categorical(y_train_, num_classes)
        y_test_ = to_categorical(y_test_, num_classes)

        
        model = create_model_task_4(num_classes,input_shape)
        model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001),
                  metrics=['accuracy'])

        history = model.fit(X_train, y_train_,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test_))
        score = model.evaluate(X_test, y_test_, verbose=0)


        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        number += 1



    print("it took", time.time() - start, "seconds.")

if __name__ == '__main__':
    # firt_task()
    # loading splitted dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_train = x_train.reshape((-1,32 * 32 * 3))
    x_test = x_test.reshape((-1,32 * 32 * 3))


    third_task(x_train, x_test, y_train, y_test)
