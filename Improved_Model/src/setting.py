import time
import os
IMG_SIZE = 124
NUM_CHANNELS=3
BATCH_SIZE = 128
EPOCHS = 100
CLASSES = ['Cat', 'Dog']
NUM_CLASSES = len(CLASSES)
cats_np_filename = "cats.npy"
dogs_np_filename = "dogs.npy"
NAME = f'dogs_vs_cats_cnn_{int(time.time())}'
MODEL_PATH = os.path.join('..','models', 'Model-36-0.889.model')
TEST_PATH = os.path.join('..','test1', '')
