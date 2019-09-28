from sklearn.model_selection import train_test_split
from define_model import define_model
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
def train(X_npyfile, y_npyfile, batch_size, epochs, num_classes):
    print("Loading data!")
    data = np.load(X_npyfile)
    labels = np.load(y_npyfile)

    print("Preprocessing data!")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=47)

    print('X_train shape:', X_train.shape)
    print('X_test shape', X_test.shape)
    print('y_train shape', y_train.shape)

    model = define_model(input_shape = X_train.shape[1:], num_classes)

    NAME = f'dogs-vs-cats-cnn-{int(time.time())}'
    filepath = "Model-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                          mode='max'))  # saves only the best ones
    log_path = 'logs/'
    log_file_name = '{}'.format(NAME)
    tensorBoard = TensorBoard(log_dir=log_path)

    callback_list = [checkpoint, tensorBoard]

    # train the neural network
    history = model.fit(X_train, y_train,
          validation_data = (X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list,
          verbose=0, shuffle = True)

    summarize_diagnostics(history)
    return history


def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['acc'], color='blue', label='train')
	pyplot.plot(history.history['val_acc'], color='orange', label='test')
	# save plot to file
	filename = 'dogs_vs_cats_cnn_training'
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
