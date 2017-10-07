from __future__ import absolute_import
from __future__ import print_function


from keras.models import Sequential
from keras.layers import Dense, Dropout


def nn_1l(model_params):
    model = Sequential()
    model.add(Dense(512, activation=model_params.activation,
                         input_shape=(model_params.input_shape,)))
    # model.add(Dropout(drop_out))
    model.add(Dense(model_params.num_classes, activation='softmax'))

    model.summary()

    return model

def nn_2l(model_params):
    model = Sequential()
    model.add(Dense(512, activation=model_params.activation,
                         input_shape=(model_params.input_shape,)))
    model.add(Dropout(model_params.drop_out))

    model.add(Dense(512, activation=model_params.activation))
    model.add(Dropout(model_params.drop_out))

    model.add(Dense(model_params.num_classes, activation='softmax'))

    model.summary()

    return model
