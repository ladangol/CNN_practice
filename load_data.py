import numpy as np
import PIL
from PIL import Image
import os, glob
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array

categories = ['dog', 'cat']

def loadimg_savenpy(path, img_size):
    X=[]
    y=[]
    for f_name in glob.glob(path+"*.jpg"):
        label = categories.index('dog')
        if f_name.startswith('cat'):
            label = categories.index('dog')
        image = Image.open(f_name)
        image = image.resize((img_size,img_size))
        image = np.asarray(image).astype('float32')/255
        X.append(image)
        y.append(label)
    X,y = np.array(X), np.array(y)
    y = keras.utils.to_categorical(y, len(np.unique(y)))       #one hot encoding
    #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 42)
    save('dogs_vs_cats_photos.npy', X)
    save('dogs_vs_cats_labels.npy', y)
