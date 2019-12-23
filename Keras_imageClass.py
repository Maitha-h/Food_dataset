from keras.datasets import cifar10
import keras.utils as utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD

# Labels
with open('C:\\Users\\Maith\\Documents\\Datasets\\Dataset_food-101\\meta\\labels.txt') as l:
    labels = l.read().splitlines()
    print(len(labels))
# Input Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
print(x_train)
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

# Model
model = Sequential()
# Adds layers in a sequential fashion one after the other
model.add(Conv2D(filters= 32, kernel_size=(7,7), input_shape=(32, 32, 3),activation='relu', padding='same', kernel_constraint= maxnorm(3)))
# outputs a feature map
model.add(MaxPooling2D(pool_size=(2, 2)))
# decreases the size of the output image
model.add(Conv2D(filters= 64, kernel_size=(5,5), input_shape=(32, 32, 3),activation='relu', padding='same', kernel_constraint= maxnorm(3)))
# outputs a feature map
model.add(MaxPooling2D(pool_size=(2, 2)))
# decreases the size of the output image
model.add(Conv2D(filters= 128, kernel_size=(3,3), input_shape=(32, 32, 3),activation='relu', padding='same', kernel_constraint= maxnorm(3)))
# outputs a feature map
model.add(MaxPooling2D(pool_size=(2, 2)))
# decreases the size of the output image
model.add(Flatten())
# converts the output into a 1D array
model.add(Dense(units=512, activation='relu', kernel_constraint=maxnorm(3)))
# creates actual prediction network
# The higher the number of units the greater the accuracy
# However, the higher the number of units the longer it takes to train
model.add(Dropout(rate=0.5))
# Drops out half the units used to increase reliability
model.add(Dense(units=10, activation='softmax'))
# number of units is the number of classes
# used to produce output for each of he 10 categories

model.compile(optimizer=SGD(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=1, batch_size=32)

# Started training the model

model.save(filepath='Image_Classifier.h5')
