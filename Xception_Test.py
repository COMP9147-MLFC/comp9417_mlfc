import os
import glob
import pandas as pd
import shutil
import cv2
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import imgaug.augmenters as iaa
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam 

# Define augmentation sequence
augmentation_seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Flipud(0.5),  # vertical flips
    iaa.Affine(rotate=(-45, 45)),  # random rotations
    iaa.GaussianBlur(sigma=(0, 3.0)),  # random Gaussian blur
    iaa.Dropout(p=(0, 0.2)),  # random drop pixels
    iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),  # random Gaussian noise
])

df = pd.read_csv('/content/Data_weed/Data - Needs Respray - 2024-03-26/Labels-NeedsRespray-2024-03-26.csv')

# Path to the directory containing the original images
original_images_dir = '/content/Data_weed/Data - Needs Respray - 2024-03-26/'

# Directory to store augmented images for 'Yes' and 'No' categories
output_dir = '/content/Data_weed/Data - Needs Respray - 2024-03-26/Augmented images'

# Create output directories if they don't exist
yes_dir = os.path.join(output_dir, 'Yes')
no_dir = os.path.join(output_dir, 'No')
os.makedirs(yes_dir, exist_ok=True)
os.makedirs(no_dir, exist_ok=True)

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    filename = row['Filename']
    needs_respray = row['Needs Respray']
    
    # Load the original image
    img_path = os.path.join(original_images_dir, filename)
    img = load_img(img_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels)
    
    # Augment the image and save the augmented images based on the 'Needs Respray' label
    if needs_respray == 'Yes':
        output_subdir = yes_dir
    else:
        output_subdir = no_dir
    
    num_augmentations = 5
    img = cv2.imread(img_path)
    images_aug = [augmentation_seq.augment_image(img) for _ in range(num_augmentations)]
    for i, img_aug in enumerate(images_aug):
        cv2.imwrite(os.path.join(output_subdir, f"Augmented_{i}_{os.path.basename(img_path)}"), img_aug)
    
# Define paths to the dataset
train_dir = '/content/Data_weed/Data - Needs Respray - 2024-03-26/Augmented images'
test_dir = '/content/Data_weed/Data - Needs Respray - 2024-03-26/'

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Prepare data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    shuffle=False)

# Load Xception model pre-trained on ImageNet
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Add custom classification layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Combine base model and custom layers
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model with the Adam optimizer and a specified learning rate
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy}")

# Predict test images
predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int).flatten()

# Get true labels from the test generator
y_true = test_generator.classes

# Calculate F1 score
f1score = classification_report(y_true, y_pred, labels=[0, 1], target_names=['No Weed', 'Weed Present'])
print(f"F1 Score:\n{f1score}")