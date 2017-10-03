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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.datasets import cifar10
from sklearn.model_selection import cross_val_score
from util import *
import time
from random import shuffle

seed = 44
rng = np.random.RandomState(seed)

# parameters
input_shape = 32 * 32
hidden_1 = 128
classes = 10
epochs = 100
batch_size = 50
lr = 0.001


def preprocessing(X):
    X_prec = []

    for item in X:
        gray_image = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)

        X_prec.append(gray_image)

    X_prec = np.array(X_prec)
    
    X_prec = X_prec.reshape((-1, 32 * 32))

    return X_prec


def batch_creator(train_x, batch_size, dataset_length, y):
    """Create batch with random samples and return appropriate format"""

    batch_mask = rng.choice(dataset_length, batch_size)

    batch_x = train_x[[batch_mask]].reshape(-1, input_shape)

    batch_x = preproc(batch_x)

    batch_y = y[[batch_mask]]
    batch_y = oneHotEncoding(batch_y)

    return batch_x, batch_y


def third_task(x_train, x_test, y_train, y_test):
    """
		
		Move on to Neural Networks, using one hidden layer. You should 
		numerically check your gradient calculations.

	"""


    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    X_ = np.concatenate((x_train, x_test))
    y_ = np.concatenate((y_train, y_test))
    y_ = [item[0] for item in y_]
    y_ = np.array(y_)

    X_ = preprocessing(X_)

    # inputs placeholders
    x = tf.placeholder(tf.float32, [None, input_shape])
    y = tf.placeholder(tf.float32, [None, classes])

    # weights/bias placeholders
    weights_bias = {
        'h1': tf.Variable(tf.random_uniform([input_shape, hidden_1])),
        'out': tf.Variable(tf.random_uniform([hidden_1, classes])),
        'b1': tf.Variable(tf.random_uniform([hidden_1])),
        'out_b1': tf.Variable(tf.random_uniform([classes]))
    }

    # constructing the graph
    # Hidden layer with RELU activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights_bias['h1']), weights_bias['b1']))

    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights_bias['out']) + weights_bias['out_b1']

    # loss definition
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
    # minimize the cost using SGD
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # init the variables
    accuracies,precisions,recalls, f1s=[],[],[],[]
    init = tf.global_variables_initializer()
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    number = 0
    start = time.time()
    for train_index, test_index in skf.split(X_, y_):

        X_train, X_test = X_[train_index], X_[test_index]
        y_train_, y_test_ = y_[train_index], y_[test_index]

        train_x = np.stack(X_train)
        print("Fold : ", number)
        with tf.Session() as sess:
            # initializing vars
            sess.run(init)

            # training
            for epoch in range(epochs):
                avg_cost = 0

                total_batch = int(len(train_x) / batch_size)
                for i in range(total_batch):
                    batch_x, batch_y = batch_creator(train_x, batch_size, train_x.shape[0], y_train_)
                    _, cost_ = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                    avg_cost += cost_ / total_batch

                print "Epoch:", (epoch + 1), "cost =", "{:.10f}".format(avg_cost)

            print "\nTraining complete!"

        # find predictions on val set
            predict = tf.argmax(out_layer, 1)
            predictions = predict.eval({x: X_test.reshape(-1, input_shape)}, session=sess)


            accuracies.append(accuracy_score(y_test_, predictions))
            precisions.append(precision_score(y_test_, predictions, average='weighted'))
            recalls.append(recall_score(y_test_, predictions, average='weighted'))
            f1s.append(f1_score(y_test_, predictions, average='weighted'))
            print("accuracy : ", accuracy_score(y_test_, predictions))
            print("precision : ", precision_score(y_test_, predictions, average='weighted'))
            print("recall : ", recall_score(y_test_, predictions, average='weighted'))
            print("f1 : ", f1_score(y_test_, predictions, average='weighted'))
            print("\n")

        number += 1

    print("Done.")

    print("accuracy avg : ", np.mean(accuracies))
    print("precision avg : ", np.mean(precisions))
    print("recall avg : ", np.mean(recalls))
    print("f1 avg : ", np.mean(f1s))

    print("it took", time.time() - start, "seconds.")


if __name__ == '__main__':
    # firt_task()
    # loading splitted dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    #x_train = x_train.reshape((-1,32 * 32 * 3))
    #x_test = x_test.reshape((-1,32 * 32 * 3))


    third_task(x_train, x_test, y_train, y_test)
