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

from __future__ import print_function



import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

from sklearn.metrics import accuracy_score
from keras.datasets import cifar10


def img2array(image):
    vector = []
    for line in image:
        for column in line:
        	vector.append(float(column[0])/255)

    return np.array(vector)

# variables

batch_size 			= 50
num_classes 		= 10
epochs 				= 200
data_augmentation 	= True
num_predictions 	= 20


# loading splitted dataset 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape((-1, 32 * 32 * 3))
x_test = x_test.reshape((-1, 32 * 32 * 3))


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

logreg = linear_model.LogisticRegression(solver='sag', max_iter=100, random_state=42,
                             multi_class='multinomial')

logreg.fit(x_train, y_train)

predictions = logreg.predict(x_test)

# print the training scores
print("training score : %.3f " % logreg.score(x_train, y_train))
print("test score : %.3f " % accuracy_score(y_test,predictions))

# results
# training score : 0.477 
# test score : 0.390 




