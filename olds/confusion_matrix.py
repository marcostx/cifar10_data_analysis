import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
# cnf_matrix = np.array([[ 0.671, 0.026, 0.03, 0.025, 0.022, 0.016, 0.026, 0.016, 0.112, 0.056],
#                      [ 0.036, 0.619, 0.01, 0.023, 0.01, 0.014, 0.025, 0.019, 0.08, 0.164],
#                      [ 0.082, 0.013, 0.405, 0.094, 0.138, 0.073, 0.102, 0.05, 0.026, 0.017],
#                      [ 0.027, 0.009, 0.06, 0.407, 0.055, 0.208, 0.116, 0.05, 0.027, 0.041],
#                      [ 0.034, 0.005, 0.099, 0.076, 0.502, 0.059, 0.102, 0.08, 0.03, 0.013],
#                      [ 0.01, 0.008, 0.055, 0.204, 0.068, 0.51, 0.05, 0.054, 0.025, 0.016],
#                      [ 0.008, 0.012, 0.045, 0.076, 0.084, 0.049, 0.681, 0.011, 0.016, 0.018],
#                      [ 0.029, 0.01, 0.029, 0.065, 0.071, 0.088, 0.024, 0.646, 0.009, 0.029],
#                      [ 0.072, 0.046, 0.008, 0.031, 0.014, 0.016, 0.008, 0.01, 0.745, 0.05 ],
#                      [ 0.041, 0.108, 0.013, 0.028, 0.011, 0.016, 0.029, 0.037, 0.054, 0.663]])

cnf_matrix = np.array([[ 0.668, 0.028, 0.044, 0.013, 0.026, 0.021, 0.035, 0.027, 0.084, 0.054],
 [ 0.024, 0.755, 0.008, 0.009, 0.003, 0.014, 0.023, 0.017, 0.023, 0.124],
 [ 0.056, 0.012, 0.492, 0.048, 0.108, 0.07, 0.149, 0.049, 0.006, 0.01 ],
 [ 0.012, 0.014, 0.063, 0.35, 0.058, 0.25, 0.187, 0.041, 0.011, 0.014],
 [ 0.025, 0.004, 0.107, 0.032, 0.527, 0.054, 0.144, 0.092, 0.009, 0.006],
 [ 0.012, 0.005, 0.042, 0.137, 0.043, 0.602, 0.101, 0.052, 0.003, 0.003],
 [ 0.005, 0.005, 0.035, 0.035, 0.048, 0.038, 0.82, 0.011, 0.002, 0.001],
 [ 0.02, 0.003, 0.024, 0.047, 0.064, 0.086, 0.027, 0.712, 0.005, 0.012],
 [ 0.08, 0.067, 0.015, 0.018, 0.022, 0.017, 0.016, 0.011, 0.697, 0.057],
 [ 0.034, 0.142, 0.013, 0.027, 0.007, 0.015, 0.028, 0.035, 0.023, 0.676]])


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], title='Confusion matrix')

plt.show()