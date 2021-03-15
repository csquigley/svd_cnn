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

class SVDConverter:
  def __init__(self,dataset,pk,print_please=False):
    self.dataset = dataset
    self.pk = pk
    self.print_please = print_please
  #new_x_train is the dataset of lowrank images created from the original cifar-10 images
  def convert(self):
    new_x_train = np.empty(self.dataset.shape)
    for top_num in range(0,len(self.dataset)):
        if self.print_please == True:
          if top_num % 1000 == 0:
              print("processed: {} of {}".format(str(top_num),str(len(self.dataset))))
        current_img = self.dataset[top_num]
        current_img = current_img.transpose()
        low_rank_img = np.empty((0,32,32))

        for i_n, layer in enumerate(current_img):

            U,s,Vt = np.linalg.svd(layer)

            #this finds the total of all the singular values, which is needed to find our percentage
            total_s = 0
            for value in s:
                total_s += value
            #current percentage
            cp = 0
            #32 is just the original image
            #anything less is a low rank approximation of the image.
            i = 1
            #find the index of the singular value that will give us the desired level of information.
            # while cp < self.pk:
            #     num = s[i]
            #     cp += (num / total_s)
            #     i += 1
            #create an array of matrices that contains the desired images
            #first create a numpy array of zeros that contains i (the number of significant singular values)
            mats = np.zeros((i,32,32))
            #reconstruct the layers from Matrix U, the singular values (s) and the Matrix V transpose
            for n in range(0,i):
                mats[n,:,:] = np.outer(U[:,n]*s[n],Vt[n,:])
            #reconstruct our image from the vectors and singular values that contain the desired information level
            reconstructed_layer = np.zeros((1,32,32))
            for n in range(0,i):
                reconstructed_layer += mats[n]

            low_rank_img = np.concatenate((low_rank_img,reconstructed_layer),axis=0)
        #the transpose gets the image back to the correct shape for an RGB image
        low_rank_img = low_rank_img.transpose()
        #add an additional dimension in order to concatenate to
        # low_rank_img = low_rank_img.reshape((1,32,32,3))
        new_x_train[top_num] = low_rank_img
        if top_num >= 50000:
          break
    return new_x_train
x_train_converter = SVDConverter(dataset=x_train,pk=0.01,print_please=False)
x_test_converter = SVDConverter(dataset=x_test,pk=0.01,print_please=False)
new_x_train = x_train_converter.convert()
new_x_test = x_test_converter.convert()
plt.imshow(new_x_train[25].astype(np.uint8))
x_test.shape
x_train_2 = new_x_train.copy()
x_test_2 = new_x_test.copy()

print(x_train.max())
print(x_test.max())
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

#this might not be wise considering that the point of this exercise is to use low rank approximations.
#doing this will competely change each matrix so that it is no longer a low rank approximation.
#especially the rotation.
generator = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
generator.fit(x_train_2)
model.fit(x_train_2,y_cat_train,batch_size=32,epochs=250,validation_data=(x_test_2,y_cat_test),callbacks=[LearningRateScheduler(lr_schedule),checkpoint])
