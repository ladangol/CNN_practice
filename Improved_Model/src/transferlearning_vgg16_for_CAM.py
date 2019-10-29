from keras.layers import GlobalAveragePooling2D, AveragePooling2D, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from setting import IMG_SIZE, NUM_CLASSES

def define_pretrained_weights_vgg16():
    #include_top = False removes the FC (fully connected) layers
    #The input shape for VGG originally has to have teh shape 224 * 224*3

    model = VGG16(include_top = False, input_shape= (IMG_SIZE, IMG_SIZE, 3))
    # don't train exsiting weights
    for layer in model.layers:
        layer.trainable = True

    #Model summary
    #print(model.summary())
    #The last layer is a Max pooling in VGG, so the second to the last is

    #Add the new classifier layers
    last_conv_layer_name = model.layers[-2].name
    gap_layer = GlobalAveragePooling2D()(model.get_layer(last_conv_layer_name).output)
    output_layer = Dense(NUM_CLASSES, activation = 'softmax')(gap_layer)

    #Define new model
    optimizer=SGD(lr=0.001, momentum=0.9)
    model = Model(inputs = model.inputs, outputs = output_layer)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
