
from keras.models import load_model
import sys, os
from sklearn.metrics import classification_report, confusion_matrix
from setting import BATCH_SIZE,CLASSES
import matplotlib.pyplot as plt
import numpy as np
import itertools


model = load_model(sys.argv[1])

def evaluate():
    print("Evaluating the model based on the best model...")
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    predictions = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = predictions.argmax(axis = 1)
    y_true = y_test.argmax(axis = 1)
    print(classification_report(y_true, y_pred, target_names = CLASSES))


    print("Building and Saving the confusion matrix...")
    confusion_mat = confusion_matrix(y_true = y_true, y_pred = y_pred)

    normalize = True
    accuracy = np.trace(confusion_mat) / float(np.sum(confusion_mat))
    misclass = 1 - accuracy
    if normalize:
        confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]


    plt.figure(figsize = (6,6))
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)

    thresh = confusion_mat.max() / 1.5 if normalize lese confusion_mat.max()/ 2
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(confusion_mat[i, j]),
                     horizontalalignment="center",
                     color="white" if confusion_mat[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(confusion_mat[i, j]),
                     horizontalalignment="center",
                     color="white" if confusion_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    confusion_matrix_plot_name = os.join.path('images', 'confusion_matrix_plot.png')
    plt.savefig(confusion_matrix_plot_name)

    plt.show()
    confusion_matrix_file_name = get_path('logs', 'confusion_matrix_file.txt')
    with open(confusion_matrix_file_name, 'w') as f:
        f.write(np.array2string(confusion_mat, separator=', '))
