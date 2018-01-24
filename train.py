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

test_rate = 0.1

img_rows, img_cols = 12, 22
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

dirs = os.listdir('./train/')

X_train, Y_train, X_test, Y_test = [], [], [], []
random.shuffle(dirs)
for dir_name in dirs:
    dir_path = 'train/%s/' % dir_name
    files = os.listdir(dir_path)
    random.shuffle(files)
    length = int(len(files) * test_rate)
    for i in files[:length]:
        Y_train.append(index[dir_name])
        image = load_img(dir_path + i, target_size=(img_rows, img_cols))
        image = img_to_array(image)
        image.resize(*input_shape)
        X_train.append(image)
    for i in files[length:length+10]:
        Y_test.append(index[dir_name])
        image = load_img(dir_path + i, target_size=(img_rows, img_cols))
        image = img_to_array(image)
        image.resize(*input_shape)
        X_test.append(image)

X_train = np.stack(X_train).astype('float32')
X_train /= 255
X_test = np.stack(X_test).astype('float32')
X_test /= 255
Y_train = np.stack(Y_train)
Y_test = np.stack(Y_test)

batch_size = 10
num_classes = 33
epochs = 100

Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

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
              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))

model.save('y2.h5')
