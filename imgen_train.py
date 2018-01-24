import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np
from PIL import Image
import os
import cv2
import random

index = {'x': 2, 'l': 11, 'j': 4, 't': 22, '5': 27, '0': 18, 'e': 5, 'k': 16, 'y': 28, 'f': 19, 's': 1, 'h': 12, 'n': 25, 'a': 23, 'm': 9, 'c': 14,
         'p': 7, 'd': 26, 'v': 31, '3': 6, '1': 8, '8': 29, '4': 13, 'r': 10, '6': 24, 'u': 20, 'q': 15, 'w': 17, '7': 21, '2': 32, 'i': 0, 'g': 30, 'b': 3}

clses = [0] * 33
for k, v in index.items():
    clses[v] = k


img_rows, img_cols = 12, 22
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


batch_size = 20
num_classes = 33
epochs = 100

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True
)
test_daragen = ImageDataGenerator(
    rescale=1. / 255
)

train_generator = train_datagen.flow_from_directory(
    'validate',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    # classes=index
)
test_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    # classes=index
)
print('data loaded')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator) // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=len(test_generator) // batch_size
)
print(train_generator.class_indices)
print(test_generator.class_indices)

predict_gen = ImageDataGenerator(
    rescale=1./255
)
predict_generator = predict_gen.flow_from_directory(
    'predict',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
)
print(model.predict_generator(
    predict_generator
))
model.save('y1.h5')
