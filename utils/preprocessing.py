from __future__ import absolute_import
from __future__ import print_function


import numpy as np


from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

MAX_PIXEL_VALUE=255

def data_augmentation(data, params):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen.fit(data)

        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # horizontal_flip=True)

def reshape(data):
    return data.reshape((-1,32 * 32 * 3))

def normalization(data):
    return data/MAX_PIXEL_VALUE

# (X - mean)/std
def padronization(data):
    return (data - data.mean())/data.std()


def processing_labels(labels, n_labels):
    labels = [item[0] for item in labels]
    labels = np.array(labels)
    return to_categorical(labels, n_labels)



def preprocessing_pipeline(x_train, x_test, y_train, y_test, n_labels, params):
    # Preprocessing data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    if params.normalization == True:
        x_train = normalization(x_train)
        x_test = normalization(x_test)

    if params.padronization == True:
        x_train = padronization(x_train)
        x_test = padronization(x_test)

    if params.data_augmentation == True:
        #print("augmentation")
        data_augmentation(x_train, params.data_augmentation_params)
        #print(x_train.shape)

    x_train = reshape(x_train)
    x_test = reshape(x_test)

    # Preprocesisng labels
    y_train = processing_labels(y_train, n_labels)
    y_test = processing_labels(y_test, n_labels)

    return x_train, x_test, y_train, y_test

