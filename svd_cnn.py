#CNN for CIFAR-10 images
#IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras import regularizers
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from converter import SVDConverter

cwd = os.getcwd()

checkpoint = ModelCheckpoint(
    filepath=cwd,
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch"
)

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    if epoch > 125:
        lrate = 0.0002
    if epoch > 150:
        lrate = 0.0001
    if epoch > 200:
        lrate = 0.00008
    if epoch > 225:
        lrate = 0.00005
    return lrate
weight_decay = 0.0001

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='elu',padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='elu',padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(.2))
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(32,32,3),activation='elu',padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(32,32,3),activation='elu',padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(.3))
model.add(Conv2D(filters=128,kernel_size=(3,3),input_shape=(32,32,3),activation='elu',padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Conv2D(filters=128,kernel_size=(3,3),input_shape=(32,32,3),activation='elu',padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(.4))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#Create Converter
x_train_converter = SVDConverter(dataset=x_train,pk=0.01,print_please=False)
x_test_converter = SVDConverter(dataset=x_test,pk=0.01,print_please=False)
#Use Converter to Convert Images
new_x_train = x_train_converter.convert()
new_x_test = x_test_converter.convert()

x_train_2 = new_x_train.copy()
x_test_2 = new_x_test.copy()
#Scale Data 1 > x > 0
x_train_2 = x_train_2/255
x_test_2 = x_test_2/255

real_x_train = x_train.copy()
real_x_test = x_test.copy()

real_x_train = real_x_train/255
real_x_test = real_x_test/255

#creates an array of all zeros except for one one to represent the category for each category
from tensorflow.keras.utils import to_categorical
y_cat_train = to_categorical(y_train,10)
y_cat_test = to_categorical(y_test,10)

# #this might not be wise considering that the point of this exercise is to use low rank approximations.
# #doing this will competely change each matrix so that it is no longer a low rank approximation.
# #especially the rotation.
# generator = ImageDataGenerator(
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
# # )
# generator.fit(x_train_2)
model.fit(x_train_2,y_cat_train,batch_size=32,epochs=250,validation_data=(x_test_2,y_cat_test),callbacks=[LearningRateScheduler(lr_schedule),checkpoint])
