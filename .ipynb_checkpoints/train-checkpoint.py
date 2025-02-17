import argparse
import os
from typing import Any

import shutil
import keras
import math
import functools
import tensorflow as tf
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import seaborn as sns
from keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img

from common import load_image_labels, create_model_densenet

IMAGE_HEIGHT = 900
IMAGE_WIDTH = 1200


########################################################################################################################

def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--train_data_labels_csv', required=True, help='Path to labels CSV')
    parser.add_argument('-t', '--target_column_name', required=True, help='Name of the column with target label in CSV')
    parser.add_argument('-o', '--trained_model_output_dir', required=True, help='Output directory for trained model')
    args = parser.parse_args()
    return args

########################################################################################################################
# IMAGE NORMALIZATION
########################################################################################################################

def normalize_images(dataset_dir: str):
    """
    Normalizes the image pixel to the range from 0 to 255
    """
    for filename in os.listdir(dataset_dir):
        if filename.endswith('g'):
            img_path = os.path.join(dataset_dir, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            #normalizing
            img_array = (img_array / np.max(img_array)) * 255
            #defining datatype of the array
            img_array = img_array.astype(np.uint8)
            #converting array to an image
            img = Image.fromarray(img_array)
            img.save(img_path)

########################################################################################################################
# ORGANIZING IMAGES BY CORRESPONDING LABELS
########################################################################################################################

def organize_images_by_label(dataset_dir: str, csv_file: str, column_name: str):

    """
    This section organizes images into a new folder according to thier label from the .csv file.
    The images are organized and stored in the following structure:
    -> resources
        -> Training
            ->Images_<TARGET_COLUMN_NAME>
                -> No <Label>
                -> Yes <Label>

    :param dataset_dir: the directory where the original training images are saved.
    :param csv_file: the directory where the csv file with labels is saved
    :param column_name: the name of the column have labels
    :returns: the training image directory
    """

    #loads the csv file
    Labels = load_image_labels(f"{dataset_dir}/{csv_file}")
    #gets a list of column names
    columns = list(Labels.columns)
    #gets a list of images
    label_names = Labels.loc[:,"Filename"]
    #performing one-hot-encoding
    expanded_labels = pd.get_dummies(Labels[column_name])
    #converting the one-hot-encoded label into boolean type
    expanded_labels = expanded_labels.astype('bool')
    #creating a new dataframe with 
    frames = [label_names, expanded_labels]
    df = pd.concat(frames, axis = 1)

    #settting up the folder names according to the labels (Yes, No)
    folder_names = Labels[column_name].unique()

    #copy images from the root directory
    copy_from = f"{dataset_dir}/"

    #copy images to the resouces directory
    copy_to = f'resources/Training/Images_{column_name}'

    os.makedirs("resources/Training", exist_ok=True)

    os.makedirs(f'resources/Training/Images_{column_name}', exist_ok=True)

    for name in folder_names:
        if os.path.exists(f'{copy_to}/{name}') == False:
            os.mkdir(f'{copy_to}/{name}')

    #Organizes and saves images
    for name in folder_names:
        files = df.Filename[df[name] == True]
        num_samples = len(files)
        num_train_samples = round(num_samples)
        i = 0
        for f in files:
            path_from = copy_from + f
            if i < num_train_samples:
                path_to = f'{copy_to}/{name}/{f}'
            shutil.copyfile(path_from, path_to)
            i += 1

    training_img_dir = copy_to

    return training_img_dir

########################################################################################################################
# STYLE TRANSFER
########################################################################################################################

    """
    THE CODE TO THIS SECTION IS REFERED FROM: https://www.tensorflow.org/tutorials/generative/style_transfer
    
    This section performs style transfer over the training images. It takes a style content image as input and creats a subtle layer of style over the original image.
    
    """

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='resources/pretrained/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    num_locations = tf.cast(tf.reduce_prod(tf.shape(input_tensor)[1:3]), tf.float32)
    return result / num_locations

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)

    def call(self, inputs):
        inputs = inputs * 255.0
        outputs = self.vgg(tf.keras.applications.vgg19.preprocess_input(inputs))
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(output) for output in style_outputs]
        content_dict = {content_name:output
                        for content_name, output in zip(self.content_layers, content_outputs)}
        style_dict = {style_name:output
                      for style_name, output in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

class StyleTransfer:
    def __init__(self, content_path, style_path, content_layers, style_layers, iterations=2, content_weight=1e4, style_weight=1e-2):
        self.content_image = load_img(content_path)
        self.style_image = load_img(style_path)
        self.iterations = iterations
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.model = StyleContentModel(style_layers, content_layers)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        self.style_targets = self.model(self.style_image)['style']
        self.content_targets = self.model(self.content_image)['content']

    def style_content_loss(self, outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name])**2) 
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.model.num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name])**2) 
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.model.num_content_layers
        return style_loss + content_loss

    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self.model(image)
            loss = self.style_content_loss(outputs)
        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    def run(self):
        image = tf.Variable(self.content_image)
        for _ in range(self.iterations):
            self.train_step(image)
        return tensor_to_image(image.read_value())
        

def image_style_transfer(training_img_dir):

    """
    Uses the functions and classes defined above to perform style transfer and save the styled images in the same directory.
    
    :param training_img_dir: the directory where the training images are saved.
    """
    content_layers = ['block5_conv2'] 
    
    style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
    
    style_path = "resources/pretrained/style_content.png"

    for classes in os.listdir(training_img_dir):
        image_dir = os.path.join(f'{training_img_dir}/', classes)
        i = 0
        for filename in os.listdir(image_dir):
            image = os.path.join(image_dir, filename)
            style_transfer = StyleTransfer(image, style_path, content_layers, style_layers)
            result_image = style_transfer.run()
            result_image_filename = f"styled_image_{i}.png"
            result_image_path = os.path.join(image_dir, result_image_filename)
            result_image.save(result_image_path)
            i+=1
    
########################################################################################################################
# IMAGE AUGMENTATIONS
########################################################################################################################

    """
    This section performs basic image augmentation techniques like rotation, flip, etc.

    The code structure of this section is:
    -> augment_and_save_dataset
        -> load_and_preprocess_image
        -> generate_multiple_augmented_sample
            -> augment_and_resize

    The augmentation technique use parallel GPU processing to generate multiple augmented samples of each image in relatively short amount of time.

    :param image_dir: the directory where the training images are saved.
    :param save_dir: the directory where the augmented images needs to be saved
    :param num_samples: number of augmented samples required for each image
    """

# Define the augmentation layers
data_augmentation_layers = tf.keras.Sequential([
    #horizontal andd vertical flip
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    #rotation
    tf.keras.layers.RandomRotation(0.2),
    #contrast
    tf.keras.layers.RandomContrast(factor=0.2),
    #Gaussian Noise
    tf.keras.layers.GaussianNoise(30)
])

def augment_and_resize(image):
    # Apply augmentations and then resize
    augmented = data_augmentation_layers(image, training=True)
    resized = tf.image.resize(augmented, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return resized
    
def load_and_preprocess_image(file_path):
    # Load, decode, and preprocess the image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH]) 
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    return image

def generate_multiple_augmented_samples(image, num_samples):
    # Generate <num_samples> augmented samples for a single image
    return tf.data.Dataset.from_tensors(image).repeat(num_samples).map(
        augment_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

def augment_and_save_dataset(image_dir, save_dir, num_samples=10):
    classes = os.listdir(image_dir)
    for class_name in classes:
        class_path = os.path.join(image_dir, class_name)
        save_class_path = os.path.join(save_dir, class_name)
        os.makedirs(save_class_path, exist_ok=True)

        dataset = tf.data.Dataset.list_files(os.path.join(class_path, '*g'), shuffle=False)
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Use flat_map to apply the generate_multiple_augmented_samples function with num_samples
        dataset = dataset.flat_map(lambda image: generate_multiple_augmented_samples(image, num_samples))
        
        # Batch and save images
        dataset = dataset.batch(1)  # Batch each image individually if saving separately
        i = 0  # Reset counter for each class
        for batch in dataset:
            save_path = os.path.join(save_class_path, f"aug_{class_name}_{i}.png")
            tf.keras.utils.save_img(save_path, batch[0].numpy())
            i += 1

########################################################################################################################
# Learning Rate Decay
########################################################################################################################

def step_decay_schedule(initial_lr, decay_factor=0.1, step_size=10):
    """
    Preforms learning rate decay by using the step-decay method.
    
    :param initial_lr: the initial learning rate
    :param decay_factor: the factor by which the learning rate should decay
    :param step_size: describe after how many steps the learning must decay
    :return: LearninigRateScheduler -> function()
    """
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))
    
    return LearningRateScheduler(schedule)

########################################################################################################################


def train(input_shape, training_data_densenet, validation_data_densenet, target_column_name, output_dir: str) -> Any:
    """
    Trains a classification model using the training images and corresponding labels.

    :param input_shape: shape of the input images for the purpose of training the model
    :param traininig_data_densenet: the data required for training the model
    :param validation_data_densenet: the data required for validate the model
    :param target_column_name: the name of the column having labels
    :param output_dir: the directory to model weights
    :return: None
    """

    #number of epochs
    n_epochs = 15
    #initial learning rate
    learning_rate = 0.001 
    #initialize decay factor
    decay_factor = 0.05
    #initialize steps for learning rate decay
    step_size = 5
    #initialize regularization strength
    reg_strength = 0.0001
    #initialize batch size
    BATCH_SIZE = 32
    #sets the number of training steps
    n_steps = training_data_densenet.samples // BATCH_SIZE
    #sets the number of validation steps
    n_val_steps = validation_data_densenet.samples // BATCH_SIZE

    #defining early stopping function to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3,
                                   verbose=1,
                                   restore_best_weights=True)
    #creates model checkpoint for the best learned weights and save the best model into the output directory
    model_checkpoint_densenet121 = ModelCheckpoint(
        filepath=f'{output_dir}/{target_column_name.replace(" ","_")}_best_weights.h5',
        save_best_only=True,
        monitor='val_loss',
        verbose=1)

    #calling the function create_model_densenet
    model_densenet = create_model_densenet(input_shape, learning_rate, reg_strength=reg_strength)
    
    #setting up the learning rate scheduler for performing step decay
    lr_scheduler = step_decay_schedule(learning_rate, decay_factor, step_size)

    #training the model
    history_densenet = model_densenet.fit(training_data_densenet,
                        validation_data=validation_data_densenet,
                        steps_per_epoch=n_steps,
                        validation_steps=n_val_steps,
                        epochs=n_epochs,
                        callbacks=[lr_scheduler, early_stopping, model_checkpoint_densenet121],
                        verbose=1)

    
    return


def main(train_input_dir: str, train_labels_file_name: str, target_column_name: str, train_output_dir: str):
    """
    The main body of the train.py responsible for
     1. loading resources
     2. loading labels
     3. loading data
     4. transforming data
     5. training model
     6. saving trained model

    :param train_input_dir: the folder with the CSV and training images.
    :param train_labels_file_name: the CSV file name
    :param target_column_name: Name of the target column within the CSV file
    :param train_output_dir: the folder to save training output.
    """
    #image normalization
    normalize_images(train_input_dir)

    #organizing images according to labels
    training_img_dir = organize_images_by_label(train_input_dir, train_labels_file_name, target_column_name)

    #peforming style transfer
    image_style_transfer(training_img_dir)

    #performing image augmentation    
    augment_and_save_dataset(training_img_dir, training_img_dir, num_samples = 30)

    #defining a fuction ImageDataGenerator to generate training and validation samples and rescale the images in range 0 to 1
    train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)

    #input shape required for DenseNet121 model
    input_shape = (224,224,3)

    #generating training data
    training_data_densenet = train_datagen.flow_from_directory(directory = training_img_dir, target_size = (224, 224), color_mode = 'rgb', batch_size = 32, class_mode = 'binary', subset='training', shuffle = True, seed = 42)

    #generating validation data
    validation_data_densenet = train_datagen.flow_from_directory(directory = training_img_dir, target_size = (224, 224), color_mode = 'rgb', batch_size = 32, class_mode = 'binary', subset='validation', shuffle = True, seed = 42)

    #performing training of the DenseNet121 model
    train(input_shape, training_data_densenet, validation_data_densenet, target_column_name, train_output_dir)

if __name__ == '__main__':
    """
    Example usage:
    
    python train.py -d "path/to/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "path/to/models"
     
    """
    args = parse_args()
    train_data_image_dir = args.train_data_image_dir
    train_data_labels_csv = args.train_data_labels_csv
    target_column_name = args.target_column_name
    trained_model_output_dir = args.trained_model_output_dir

    main(train_data_image_dir, train_data_labels_csv, target_column_name, trained_model_output_dir)

########################################################################################################################
