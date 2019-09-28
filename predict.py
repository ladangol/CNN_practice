
from keras.models import load_model

# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import csv
import os, glob
import numpy as np

categories = ['cat', 'dog']

# load and prepare the image
def load_image(filename, img_size):
	# load the image
	img = load_img(filename, target_size=(img_size, img_size))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, img_size, img_size, 3)
	# center pixel data
	img = img.astype('float32')
	return img

# load an image and predict the class
def predict(model_path, testfolder_path, img_size):
    model = load_model(model_path)
    for img_p in glob.glob(testfolder_path+"*.jpg"):
        img= load_image(img_p, img_size)
        pred = model.predict(img.reshape(-1, img_size, img_size, 3))
        print(pred)
        classId = np.argmax(pred)
        className = categories[classId]
        path, image_name_w_ext = os.path.split(img_p)
        image_name, image_ext = os.path.splitext(image_name_w_ext)
        print(image_name_w_ext + ': Prediction ' + className)
        #plt.imshow(img, cmap=plt.cm.binary)
        #plt.xlabel(className)
        #plt.show()
        row = [image_name, classId]
        with open('results.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
