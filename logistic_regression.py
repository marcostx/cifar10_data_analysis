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
import numpy as np
from sklearn import linear_model

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.datasets import cifar10
from sklearn.model_selection import cross_val_score

import utils.preprocessing as pp



def first_task(x_train,x_test,y_train,y_test):
	"""

		Perform Logistic Regression as the baseline (first solution) to learn
		the 10 classes in the dataset. Use one-vs-all strategy to build a
		classification model.

	"""

	#x_train = StandardScaler().fit_transform(x_train)
	#x_test =  StandardScaler().fit_transform(x_test)


	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	logreg = linear_model.LogisticRegression(solver='lbfgs', max_iter=100, random_state=42,verbose=1,
	                             multi_class='ovr')

	#logreg.fit(x_train, y_train)

	#predictions = logreg.predict(x_test)

	# print the training scores
	#print("training score : %.3f " % logreg.score(x_train, y_train))
	#print("test score : %.3f " % accuracy_score(y_test,predictions))

	# results
	# training score : 0.477

	# using cross validation
	#logreg = linear_model.LogisticRegression(solver='lbfgs', max_iter=100, random_state=42,verbose=1,
	#                             multi_class='multinomial')


	X = np.concatenate((x_train,x_test))
	y = np.concatenate((y_train,y_test))
	y = [item[0] for item in y]

	print(cross_val_score(logreg, X, y, cv=10,verbose=True))

	# cv results
	#[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  5.3min finished
	#[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  5.0min finished
	#[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  5.2min finished
	#[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  5.6min finished
	#[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  5.4min finished
	#[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  5.1min finished
	#[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  5.3min finished
	#[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  5.4min finished
	#[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  5.4min finished
	#[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  5.2min finished
	#[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed: 53.3min finished
	#[ 0.38883333  0.39416667  0.37966667  0.39983333  0.39183333  0.38566667
	#  0.38883333  0.38666667  0.38966667  0.38333333]


def second_task(x_train,x_test,y_train,y_test):
	"""

		Perform Multinomial Logistic Regression (i.e., Softmax regression).
		It is a generalization of Logistic Regression to the case where we
		want to handle multiple classes

	"""

	#x_train = StandardScaler().fit_transform(x_train)
	#x_test =  StandardScaler().fit_transform(x_test)


	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	logreg = linear_model.LogisticRegression(solver='lbfgs', max_iter=100, random_state=42,verbose=1,
	                             multi_class='multinomial')

	#logreg.fit(x_train, y_train)

	#predictions = logreg.predict(x_test)

	# print the training scores
	#print("training score : %.3f " % logreg.score(x_train, y_train))
	#print("test score : %.3f " % accuracy_score(y_test,predictions))

	# results
	# training score : 0.477

	# using cross validation
	#logreg = linear_model.LogisticRegression(solver='lbfgs', max_iter=100, random_state=42,verbose=1,
	#                             multi_class='multinomial')


	X = np.concatenate((x_train,x_test))
	y = np.concatenate((y_train,y_test))
	y = [item[0] for item in y]

	print(cross_val_score(logreg, X, y, cv=10,verbose=True))

	# cv results
	#[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  1.8min finished
	#[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  2.0min finished
	#[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  2.0min finished
	#[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  1.8min finished
	#[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  1.9min finished
	#[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  2.1min finished
	#[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  2.3min finished
	#[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  2.3min finished
	#[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  1.8min finished
	#[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:  1.8min finished
	#[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed: 20.2min finished
	#[ 0.399       0.39733333  0.38966667  0.411       0.408       0.40233333
	#  0.39916667  0.40366667  0.40883333  0.39316667]

if __name__ == '__main__':
	#firt_task()
	# loading splitted dataset
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	#x_train = x_train.reshape((-1, 32 * 32 * 3))
	#x_test = x_test.reshape((-1, 32 * 32 * 3))
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	x_train = pp.normalization(x_train)
	x_test = pp.normalization(x_test)

	x_train = pp.padronization(x_train)
	x_test = pp.padronization(x_test)

	x_train = pp.reshape(x_train)
	x_test = pp.reshape(x_test)

	first_task(x_train,x_test,y_train,y_test)



