from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from matplotlib import pyplot
from define_model import define_model
import time
import os,sys
from setting import BATCH_SIZE, EPOCHS, NUM_CLASSES, cats_np_filename, dogs_np_filename,NAME
from transferlearning_vgg16_for_CAM import define_pretrained_weights_vgg16


def summarize_diagnostics(history):
    #plot loss
    pyplot.subplot(2,1,1)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color = 'blue', label = 'train')
    pyplot.plot(history.history['val_loss'], color = 'orange', label = 'test')
    #plot accuracy
    pyplot.subplot(2,1,2)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['acc'], color = 'blue', label = 'train')
    pyplot.plot(history.history['val_acc'], color = 'orange', label = 'test')

    pyplot.tight_layout()
    #Save plot to a file
    filename = os.path.join('..','images','dogs_vs_cats_cnn_training')
    pyplot.savefig(filename+'_plot.png')
    pyplot.close()

def train(transfer_or_not):
    cats = np.load(cats_np_filename)
    dogs = np.load(dogs_np_filename)
    print(EPOCHS)
    #adding the label in the one-hot encoding
    #According to kaggle 1= dog, 0 = cat
    #Recal [5]*3 is [5,5,5]
    #Recal cats.shape[0] gives us the number of cat images
    cats_y = np.array([[1,0]]*cats.shape[0])
    dogs_y = np.array([[0,1]]*dogs.shape[0])

    print("cats_y", cats_y.shape)
    print("dogs_y", dogs_y.shape)

    X =  np.concatenate([cats, dogs], axis = 0)
    y = np.concatenate([cats_y, dogs_y], axis = 0)

    print("X shape", X.shape)
    print("y shape", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle = "True", random_state = 42)
    #np.save('X_test.npy', X_test)
    #np.save('y_test.npy', y_test)

    if not transfer_or_not:
        model = define_model(input_shape=X_train.shape[1:], num_classes= NUM_CLASSES)
    else:
        print("In VGG")
        model = define_pretrained_weights_vgg16()
        #EPOCHS = 200
    #unique file name that will include the epoch and the validation acc for that epoch
    filepath = "Model-{epoch:02d}-{val_acc:.3f}"
    #saves only the best one at each epoch
    checkpoint = ModelCheckpoint(os.path.join('..','models','{}.model').format(filepath, monitor = 'val_loss',
    verbos = 1, save_best_only=True, mode = 'min'))


    log_file_name = '{}'.format(NAME)
    log_path = os.path.join('..','logs', log_file_name)
    tensorboard = TensorBoard(log_dir = log_path)

    csv_log_name = 'dogs_vs_cats_history_log_vgg.csv' if transfer_or_not else 'dogs_vs_cats_history_log.csv'
    csv_logger = CSVLogger(os.path.join('..','logs', csv_log_name), append = True)

    callback_list = [checkpoint, tensorboard, csv_logger]

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = EPOCHS, batch_size=BATCH_SIZE, callbacks = callback_list, verbose = 1, shuffle=True)

    summarize_diagnostics(history)

transfer_or_not =sys.argv[1] == "True"
#assert isinstance(transfer_or_not, bool),
#raise TypeError('param should be a bool')

print('Start Training....')
#print(str(transfer_or_not))
train(transfer_or_not)
