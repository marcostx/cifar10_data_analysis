from __future__ import absolute_import
from __future__ import print_function


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU



def nn_1l(model_params):
    activation = model_params.activation
    if activation == 'leakyrelu':
        activation = 'linear'

    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(model_params.l1_size, activation=activation))
    if model_params.activation == 'leakyrelu':
        model.add(LeakyReLU(alpha=.001))
    # model.add(Dropout(drop_out))
    model.add(Dense(model_params.num_classes, activation='softmax'))

    model.summary()

    return model

def nn_2l(model_params):
    activation = model_params.activation
    if activation == 'leakyrelu':
        activation = 'linear'

    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(model_params.l1_size, activation=activation))
    if model_params.activation == 'leakyrelu':
        model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(model_params.drop_out))

    model.add(Dense(model_params.l2_size, activation=activation))
    if model_params.activation == 'leakyrelu':
        model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(model_params.drop_out))

    model.add(Dense(model_params.num_classes, activation='softmax'))

    model.summary()

    return model
