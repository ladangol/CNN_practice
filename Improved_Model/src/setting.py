import time
IMG_SIZE = 124
BATCH_SIZE = 64
EPOCHS = 100
CLASSES = ['Dog', 'Cat']
NUM_CLASSES = len(CLASSES)
cats_np_filename = "cats.npy"
dogs_np_filename = "dogs.npy"
NAME = f'dogs_vs_cats_cnn_{int(time.time())}'
