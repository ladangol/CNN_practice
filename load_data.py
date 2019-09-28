import numpy as np
import PIL
from PIL import Image
import os, glob
from random import shuffle
#import keras
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array

categories = ['cat', 'dog']

def loadimg_savenpy(path, img_size):
    X=[]
    y=[]
    files = glob.glob(path+"*.jpg")
    shuffle(files)
    for f_name in files:
        label = categories.index('dog')
        if f_name.find('cat') != -1:
            label = categories.index('cat')
        image = Image.open(f_name)
        image = image.resize((img_size,img_size))
        image = np.asarray(image).astype('float32')/255
        X.append(image)
        y.append(label)
        print(y)
    X,y = np.array(X), np.array(y)
    #randomize = np.arange(len(X))
    #np.random.shuffle(randomize)
    #X = X[randomize]
    #y = y[randomize]
    #Problem: even if I change to categorical here when I load the npy file it is (num_images, 1) dim instead of (num_images_2)
    #y = keras.utils.to_categorical(y, len(np.unique(y)))       #one hot encoding
    #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 42)
    np.save('dogs_vs_cats_photos.npy', X)
    np.save('dogs_vs_cats_labels.npy', y)
