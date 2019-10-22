from keras.models import load_model
import matplotlib.pyplot as plt
import csv
import os, glob
import numpy as np
from setting import MODEL_PATH, TEST_PATH, IMG_SIZE, CLASSES
from load_data import import_image

def predict(model_path, testfolder_path, img_size):
    model = load_model(model_path)
    print(model.summary)
    for img_p in glob.glob(testfolder_path+"*.jpg"):
        img= import_image(img_p)
        pred = model.predict(img.reshape(-1, img_size, img_size, 3))
        print(pred)
        classId = np.argmax(pred)
        className = CLASSES[classId]
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


predict(MODEL_PATH,TEST_PATH, IMG_SIZE)
