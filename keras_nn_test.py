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
import numpy as np
import cv2
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.datasets import cifar10
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import time
from keras.utils import to_categorical
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

# Local imports
import utils.load_config as lc
import argparse


def preproc(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen.fit(x_train)

    x_train = x_train.reshape((-1,32 * 32 * 3))
    x_test = x_test.reshape((-1,32 * 32 * 3))

    return x_train,x_test

def nn_1l(num_classes,input_shape, activation, drop_out):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(input_shape,)))
    # model.add(Dropout(drop_out))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    return model

def nn_2l(num_classes,input_shape, activation, drop_out):
    model = Sequential()
    model.add(Dense(512, activation=activation, input_shape=(input_shape,)))
    model.add(Dropout(drop_out))
    model.add(Dense(512, activation=activation))
    model.add(Dropout(drop_out))
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


def train(dataset, num_classes, input_shape, batch_size, epochs, learning_rate, model_function, activation, drop_out):
    x_train = dataset[0]
    y_train = dataset[1]
    x_test = dataset[2]
    y_test = dataset[3]

    print('x_train shape:', x_train.shape)
    print(x_train.shape, 'train samples')
    print(x_test.shape, 'test samples')

    y_train = [item[0] for item in y_train]
    y_train = np.array(y_train)
    y_train = to_categorical(y_train, num_classes)

    y_test = [item[0] for item in y_test]
    y_test = np.array(y_test)
    y_test = to_categorical(y_test, num_classes)

    model = model_function(num_classes, input_shape, activation, drop_out)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])

    start = time.time()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)
    print("it took", time.time() - start, "seconds.")

    y_predicted = model.predict(x_test, verbose=0)

    y_test_ = np.argmax(y_test, axis=1)
    y_predicted = np.argmax(y_predicted, axis=1)

    cm = confusion_matrix(y_test_, y_predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    accuracy = np.mean(np.diag(cm))

    print('confusion matrix:', cm)
    print('Accuracy of cm:', accuracy)


dispatcher = {
    'nn_1l':nn_1l,
    'nn_2l':nn_2l
}

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Config file', required=True)
    args = parser.parse_args()

    CONFIG = lc.load_configuration(args.config_file)


    num_classes   = int(CONFIG.num_classes)
    channels      = int(CONFIG.channels)
    img_size      = int(CONFIG.img_size)
    batch_size    = int(CONFIG.batch_size)
    epochs        = int(CONFIG.epochs)
    learning_rate = float(CONFIG.learning_rate)
    model         = CONFIG.model

    drop_out      = float(CONFIG.drop_out)
    activation    = CONFIG.activation

    input_shape   = (img_size**2) * channels

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train,x_test = preproc(x_train,x_test)

    dataset = [x_train, y_train, x_test, y_test]


    train(dataset, num_classes, input_shape, batch_size, epochs, learning_rate, dispatcher[model], activation, drop_out)




if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))