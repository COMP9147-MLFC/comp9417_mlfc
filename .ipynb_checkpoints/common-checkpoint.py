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
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    max_val = tf.reduce_max(img)
    image_norm = (image/max_val)*255
    image = image_norm.numpy().astype("uint8")
    

    # The following are examples on how you might manipulate the image.
    # See full documentation on Pillow (PIL): https://pillow.readthedocs.io/en/stable/

    # To make the image 50% smaller
    # Determine image dimensions
    # width, height = image.size
    # new_width = int(width * 0.50)
    # new_height = int(height * 0.50)
    # image = image.resize((new_width, new_height))

    # To crop the image
    # (left, upper, right, lower) = (20, 20, 100, 100)
    # image = image.crop((left, upper, right, lower))

    # To view an image
    # image.show()

    # Return either the pixels as array - image_array
    # To convert to a NumPy array
    # image_array = np.asarray(image)
    # return image_array

    # or return the image
    return image



########################################################################################################################
# Model Loading and Saving Functions
########################################################################################################################

def create_model_densenet(input_shape, lr, reg_strength=0.3):
    opt = SGD(learning_rate=lr)
    
    conv_base = DenseNet121(include_top=False,
                      weights='imagenet',
                      input_shape=input_shape)
    
    for layer in conv_base.layers:
        layer.trainable = False
        
    top_model = conv_base.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(128, activation='relu', kernel_regularizer=l2(reg_strength))(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dropout(0.5)(top_model)
    output_layer = Dense(1, activation='sigmoid')(top_model)
    
    model = Model(inputs=conv_base.input, outputs=output_layer)
    
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def save_model(model: Any, target: str, output_dir: str):
    """
    Given a model and target label, save the model file in the output_directory.

    The specific format depends on your implementation.

    Common Deep Learning Model File Formats are:

        SavedModel (TensorFlow)
        Pros: Framework-agnostic format, can be deployed in various environments. Contains a complete model representation.
        Cons: Can be somewhat larger in file size.

        HDF5 (.h5) (Keras)
        Pros: Hierarchical structure, good for storing model architecture and weights. Common in Keras.
        Cons: Primarily tied to the Keras/TensorFlow ecosystem.

        ONNX (Open Neural Network Exchange)
        Pros: Framework-agnostic format aimed at improving model portability.
        Cons: May not support all operations for every framework.

        Pickle (.pkl) (Python)
        Pros: Easy to save and load Python objects (including models).
        Cons: Less portable across languages and environments. Potential security concerns.

    IMPORTANT: Add additional arguments as needed. We have just given the model as an argument, but you might have
    multiple files that you save.

    :param model: the model that you want to save.
    :param target: the target value - can be useful to name the model file for the target it is intended for
    :param output_dir: the output directory to same one or more model files.
    """
    # TODO: implement your model saving code here
    raise RuntimeError("save_model() is not implemented.")


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

    model_densenet = create_model_densenet(input_shape_densenet, learning_rate, reg_strength=reg_strength)

    model = model_densenet.load_weights(f'{trained_model_dir}/{target_column_name.replace(" ","_")}_best_weights.h5')
    
    raise RuntimeError("load_model() is not implemented.")
    return model