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
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.datasets import cifar10
from sklearn.model_selection import cross_val_score
import time
from random import shuffle

seed = 44
rng = np.random.RandomState(seed)


def third_task(x_train,x_test,y_train,y_test):
	"""
		
		Move on to Neural Networks, using one hidden layer. You should 
		numerically check your gradient calculations.

	"""

	#x_train = StandardScaler().fit_transform(x_train)
	#x_test =  StandardScaler().fit_transform(x_test)


	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	X_ = np.concatenate((x_train,x_test))
	y_ = np.concatenate((y_train,y_test))
	y_ = [item[0] for item in y_]

	# parameters
	input_shape = 32*32
	hidden_1 	= 128
	classes     = 10 
	epochs 		= 150
	batch_size  = 50
	lr 			= 0.001


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
	layer_1 =  tf.nn.relu(tf.add(tf.matmul(x, weights_bias['h1']), weights_bias['b1']))


	# Output layer with linear activation
	out_layer = tf.matmul(layer_1, weights_bias['out']) + weights_bias['out_b1']
    
    # loss definition
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

	# minimize the cost using SGD
	optimizer.minimize(cost)

	# init the variables
	init = tf.global_variables_initializer()
	skf = StratifiedKFold(n_splits=10, shuffle=True)
	number = 0
	start = time.time()
	for train_index, test_index in skf.split(X_, y_):
	    print(y_)

	    X_train, X_test = X_[train_index], X_[test_index]
	    y_train, y_test = y_[train_index], y_[test_index]
	    
	    train_x = np.stack(X_train)
	    print("Fold : ", number)

	print("it took", time.time() - start, "seconds.")











	

	

if __name__ == '__main__':
	#firt_task()
	# loading splitted dataset 
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	x_train = x_train.reshape((-1,32 * 32 * 3))
	x_test = x_test.reshape((-1,32 * 32 * 3))

	third_task(x_train,x_test,y_train,y_test)



