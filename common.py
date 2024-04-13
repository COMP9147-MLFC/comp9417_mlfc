from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import 

import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from os import listdir
from os.path import isfile, join

def get_image(url):
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


########################################################################################################################
# Data Augmentation functions
########################################################################################################################
def ai_gen_augment_1(image_file_path: str):
    """
    Uses a pre-trained Generative AI Algorithm to create additional virtual samples based on original images
    :param image_file_path: the path to image file.
    :return img_aug: the augmented image
    """

    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    prompt = "same image"

    im = get_image(image_final_path)
    images = pipe(prompt, image=im, num_inference_steps=10, image_guidance_scale=1).images

    return images[0]

def image_augment(image_file_path: str):
    """
    Uses various image transformations such as rotations, flips and zooms to create additional virtual samples
    :param image_file_path: the path to image file.
    :return img_aug: the augmented image
    """
    # TODO Implement Image Manipulation Augment Function
    return img_aug

def noise_augment(image_file_path: str):
    """
    Applies random gaussian noise to obfuscate features in the image and generate additional virtual samples
    :param image_file_path: the path to image file.
    :return img_aug: the augmented image
    """
    
    img = cv2.imread(image_file_path)[...,::-1]/255.0
    noise =  np.random.normal(loc=0, scale=1, size=img.shape)
    img_aug = np.clip((img*(1 + noise*0.4)),0,1)

    return img_aug


########################################################################################################################
# Data Loading functions
########################################################################################################################
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
    # TODO: implement your model loading code here
    raise RuntimeError("load_model() is not implemented.")
