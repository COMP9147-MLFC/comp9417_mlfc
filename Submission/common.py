from typing import Any

import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout


def load_image_labels(labels_file_path: str):
    """
    Loads the labels from CSV file.

    :param labels_file_path: CSV file containing the image and labels.
    :return: Pandas DataFrame
    """
    df = pd.read_csv(labels_file_path)
    return df


def load_predict_image_names(predict_image_list_file: str) -> [str]:
    """
    Reads a text file with one image file name per line and returns a list of files
    :param predict_image_list_file: text file containing the image names
    :return list of file names:
    """
    with open(predict_image_list_file, 'r') as file:
        lines = file.readlines()
    # Remove trailing newline characters if needed
    lines = [line.rstrip('\n') for line in lines]
    return lines


def load_single_image(image_file_path: str) -> Image:
    """
    Load the image.

    NOTE: you can optionally do some initial image manipulation or transformation here.

    :param image_file_path: the path to image file.
    :return: Image (or other type you want to use)
    """
    # Load the image
    image = Image.open(image_file_path)
    
    return image



########################################################################################################################
# Model Creating, Loading and Saving Functions
########################################################################################################################

def create_model_densenet(input_shape, lr, reg_strength=0.3):

    """
    Creates DenseNet121 model with pretrainined imagenet weights and excluding the top classification layer. 

    :param image_file_path: the path to image file.
    :return: Image (or other type you want to use)
    """
    
    opt = SGD(learning_rate=lr)

    #loads pretrained imagenet weights from the local directory
    conv_base = DenseNet121(include_top=False,
                      weights='resources/pretrained/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                      input_shape=input_shape)
    
    for layer in conv_base.layers:
        layer.trainable = False

    # setting up model architechture
    top_model = conv_base.output
    # adding a Global Average Pooling layer to conver the 4D tensor output of the conv_base into a 2D tensor
    top_model = GlobalAveragePooling2D()(top_model)
    # adding a dense layer with 128 neurons, ReLU activation function, L2 regularization, and batch normalization
    top_model = Dense(128, activation='relu', kernel_regularizer=l2(reg_strength))(top_model)
    top_model = BatchNormalization()(top_model)
    # adding a dropout layer with a dropout rate of 50%
    top_model = Dropout(0.5)(top_model)
    # defining the final output layer with activation function sigmoid for our binary classification problem. The result will the probabilies.
    output_layer = Dense(1, activation='sigmoid')(top_model)

    #creating the modified model
    model = Model(inputs=conv_base.input, outputs=output_layer)
    
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model


def load_model(trained_model_dir: str, target_column_name: str) -> Any:
    """
    Given a model and target label, save the model file in the output_directory.

    The specific format depends on your implementation and should mirror save_model()

    IMPORTANT: Add additional arguments as needed. We have just given the model as an argument, but you might have
    multiple files that you save.

    :param trained_model_dir: the directory where the model file(s) are saved.
    :param target_column_name: the target value - can be useful to name the model file for the target it is intended for
    :returns: the model
    """
    input_shape = (224, 224, 3)
    learning_rate = 0.0001

    model_densenet = create_model_densenet(input_shape, learning_rate, reg_strength=0.3)
    
    #loading weights
    model_densenet.load_weights(f'{trained_model_dir}/{target_column_name.replace(" ","_")}_best_weights.h5')
    
    return model_densenet