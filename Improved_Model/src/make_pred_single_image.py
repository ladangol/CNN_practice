from keras.models import load_model, Model
import sys, os
import numpy as np
from PIL import Image
from setting import IMG_SIZE, CLASSES
import matplotlib.pyplot as plt
import scipy as sp

model = load_model(sys.argv[1])
image_path = str(sys.argv[2])
last_conv_layer_name = 'conv2d_6'

def make_prediction(model, image_path, last_conv_layer_name):
    gap_weights = model.layers[-1].get_weights()[0]
    image = np.array(Image.open(image_path).resize((IMG_SIZE,IMG_SIZE)), dtype = "float32")/255.
    #image shape is IMG_SIZE*IMG_SIZE*3
    """
    The type of model.input and each layer output is a tensor
    as an example for model input
        Tensor("conv2d_1_input_1:0", shape=(?, 124, 124, 3), dtype=float32)
    and for a layer output:
        Tensor("conv2d_6_1/Relu:0", shape=(?, 23, 23, 128), dtype=float32)
    """
    cam_model = Model(inputs= model.input, outputs =(model.get_layer(last_conv_layer_name).output, model.layers[-1].output))
    #cam_model.summary()
    """
    Note that the cam model is same as the model but it has two outputs: along with the final predictions
    We have the features from the last conv layer

    cam_model has two outputs as specified above
    Note that reshaping the image to a 4 dim array is necessary
    because conv2d_1_input (the input of first layer of CNN) expects a 4 dim array
    """
    features, result = cam_model.predict(image.reshape(-1, IMG_SIZE, IMG_SIZE, 3))

    #features shape is (1, 23, 23, 128)
    #to igonre the first dim adn get the image
    features_for_one_image = features[0,:,:,:]

    height_roomout = IMG_SIZE / features_for_one_image.shape[0]
    width_roomout = IMG_SIZE / features_for_one_image.shape[1]

    cam_features = sp.ndimage.zoom(features_for_one_image, (height_roomout, width_roomout, 1), order=2)

    pred_index = result.argmax()
    print(result)
    cam_weights = gap_weights[:, pred_index]
    cam_output = np.dot(cam_features, cam_weights)

    pred_label = CLASSES[pred_index]
    plt_label = 'Predicted Class:{},Probability:{}'.format(pred_label,str(result[0][pred_index]))
    plt.figure(facecolor='white')
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(122)
    plt.imshow(image)
    plt.imshow(cam_output, cmap='jet', alpha=0.5)
    plt.suptitle(plt_label, fontsize=10)
    plt.title('Class Activation Map')
    plt.show()

make_prediction(model, image_path, last_conv_layer_name)
