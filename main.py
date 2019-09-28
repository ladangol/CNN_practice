from load_data import loadimg_savenpy
from train import train
IMG_SIZE = 128
num_classes = 2
epochs = 200
batch_size = 32

do_loadimage = False
do_train = True

if do_loadimage:
    loadimg_savenpy('train/', IMG_SIZE)
if do_train:
    train('dogs_vs_cats_photos.npy', 'dogs_vs_cats_labels.npy', batch_size, epochs, num_classes)
