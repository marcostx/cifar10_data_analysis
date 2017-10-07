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



import time
import os
import sys
import argparse
import numpy as np

import tensorflow as tf

from keras.datasets import cifar10

from sklearn.metrics import confusion_matrix

# Local imports
import utils.train_models as tm
import utils.load_config as lc
import utils.preprocessing as pp




def test_pipeline(dataset, model_params, network_params):
    # Split dataset
    x_train = dataset[0]
    y_train = dataset[1]
    x_test = dataset[2]
    y_test = dataset[3]

    # Train model
    model = tm.train(x_train, y_train, network_params, model_params)

    # Test model
    y_predicted = model.predict(x_test, verbose=0)

    # Get comfusion matrix
    y_test_ = np.argmax(y_test, axis=1)
    y_predicted = np.argmax(y_predicted, axis=1)

    cm = confusion_matrix(y_test_, y_predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Get accuracy
    accuracy = np.mean(np.diag(cm))

    print('confusion matrix:', cm)
    print('Accuracy of cm:', accuracy)


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

    # Get dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocessing dataset
    x_train, x_test, y_train, y_test = pp.preprocessing_pipeline(x_train, x_test, y_train, y_test, model_params.num_classes, preprocessing_params)

    dataset = [x_train, y_train, x_test, y_test]


    test_pipeline(dataset, model_params, network_params)




if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))