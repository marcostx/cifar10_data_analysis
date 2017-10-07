from __future__ import absolute_import
from __future__ import print_function


from keras.optimizers import Adam

import utils.models as models


dispatcher = {
    'nn_1l':models.nn_1l,
    'nn_2l':models.nn_2l
}


def train(x_train, y_train, network_params, model_params, datagen):

    model = dispatcher[network_params.model](model_params)

    model.compile(loss=network_params.loss,
                  optimizer=Adam(lr=network_params.learning_rate),
                  metrics=['accuracy'])

    # start = time.time()

    if datagen != None:
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=network_params.batch_size),
                            steps_per_epoch=x_train.shape[0]//network_params.batch_size,
                            epochs=network_params.epochs,
                            verbose=network_params.verbose)
    else:
        history = model.fit(x_train, y_train,
                            batch_size=network_params.batch_size,
                            epochs=network_params.epochs,
                            verbose=network_params.verbose)
    print(history)
    # print("it took", time.time() - start, "seconds.")

    return model