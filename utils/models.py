from __future__ import absolute_import
from __future__ import print_function


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten


def nn_1l(model_params):
    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(model_params.l1_size, activation=model_params.activation))
    # model.add(Dropout(drop_out))
    model.add(Dense(model_params.num_classes, activation='softmax'))

    model.summary()

    return model

def nn_2l(model_params):
    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    model.add(Dense(model_params.l1_size, activation=model_params.activation))
    model.add(Dropout(model_params.drop_out))

    model.add(Dense(model_params.l2_size, activation=model_params.activation))
    model.add(Dropout(model_params.drop_out))

    model.add(Dense(model_params.num_classes, activation='softmax'))

    # model.summary()

    return model
