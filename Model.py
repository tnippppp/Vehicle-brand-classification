# Import libraries
from keras import layers
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
import cv2 as cv
import os
import numpy as np
from datetime import datetime

# Prepare data
train_path = "C:\\Users\\wanwisa.j\\Documents\\GitHub\\Vehicle-classification\\archive\\Cars Dataset\\train"
test_path = "C:\\Users\\wanwisa.j\\Documents\\GitHub\\Vehicle-classification\\archive\\Cars Dataset\\test"
valid_path = "C:\\Users\\wanwisa.j\\Documents\\GitHub\\Vehicle-classification\\archive\\Cars Dataset\\valid"

BATCH_SIZE = 32
IMAGE_SIZE = 224
CLASS_SIZE = 7

train_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]
train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical')

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
    directory=valid_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical')

# Check image and class
class_names = list(train_generator.class_indices.keys())  # Get the class names from the generator
class_names

# Create model
from keras.applications import MobileNetV3Small

base_model = MobileNetV3Small(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = True

inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(CLASS_SIZE, activation='softmax')(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
custom_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.001,
    mode='min'
)

# Train model
start = datetime.now()

history = model.fit(
    train_generator,
    epochs=3,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    callbacks=[custom_early_stopping]
)

print('Execution Time:', datetime.now() - start)

model.summary()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Create model_0
model_0 = model

# Create class_names
class_names = list(train_generator.class_indices.keys())