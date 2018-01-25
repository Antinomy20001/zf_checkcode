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

index = {'0': 0,'1': 1,'2': 2,'3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'a': 9,'b': 10,'c': 11,'d': 12,'e': 13,'f': 14,'g': 15,'h': 16,'i': 17,'j': 18,'k': 19,'l': 20,'m': 21,'n': 22,'p': 23,'q': 24,'r': 25,'s': 26,'t': 27,'u': 28,'v': 29,'w': 30,'x': 31,'y': 32}

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
    'train',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    # classes=index
)
test_generator = train_datagen.flow_from_directory(
    'validate',
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

model.save('ok.h5')
