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



import sys
import argparse
import numpy as np

import tensorflow as tf

from keras.datasets import cifar10
import keras.backend.tensorflow_backend as ktf


from sklearn.model_selection import StratifiedKFold


# Local imports
import utils.train_models as tm
import utils.load_config as lc
import utils.preprocessing as pp
import utils.tf_session as tfs


def train_pipeline(preprocessing_params, model_params, network_params):

    # Get dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocessing dataset
    x_train, x_test, y_train, y_test, datagen = pp.preprocessing_pipeline(x_train, x_test, y_train, y_test, model_params.num_classes, preprocessing_params)

    # Split dataset
    y_train_ = np.argmax(y_train, axis=1)

    # init the variables
    accuracies=[]

    skf = StratifiedKFold(n_splits=network_params.n_kfolds, shuffle=True)
    for train_index, val_index in skf.split(x_train, y_train_):
        kf_x_train, kf_y_train = x_train[train_index], y_train[train_index]
        kf_x_test, kf_y_test   = x_train[val_index], y_train[val_index]

        # Train model
        model = tm.train(kf_x_train, kf_y_train, network_params, model_params, datagen)

        score = model.evaluate(kf_x_test, kf_y_test, verbose=0)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        accuracies.append(score[1])


    print("avg accuracy cv=", network_params.n_kfolds, ":", np.mean(accuracies))

    return np.mean(accuracies)

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Config file', required=True)
    args = parser.parse_args()

    CONFIG = lc.load_configuration(args.config_file)

    # Get parameters
    model_params = CONFIG.model_params
    network_params = CONFIG.network_params
    preprocessing_params = CONFIG.preprocessing_params

    ktf.set_session(tfs.get_session(network_params.gpu_fraction))

    train_pipeline(preprocessing_params, model_params, network_params)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))