from __future__ import absolute_import
from __future__ import print_function


from keras.optimizers import Adam

import utils.models as models


dispatcher = {
    'nn_1l':models.nn_1l,
    'nn_2l':models.nn_2l
}


def train(x_train, y_train, network_params, model_params):

    model = dispatcher[network_params.model](model_params)

    model.compile(loss=network_params.loss,
                  optimizer=Adam(lr=network_params.learning_rate),
                  metrics=['accuracy'])

    # start = time.time()
    history = model.fit(x_train, y_train,
                        batch_size=network_params.batch_size,
                        epochs=network_params.epochs,
                        verbose=network_params.verbose)
    # print("it took", time.time() - start, "seconds.")

    return model