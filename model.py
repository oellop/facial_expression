
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.utils import shuffle
import tensorflow as tf
import keras
# from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
import pandas as pd

from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D

train = []
label = []


def data_prep_CK1():
    path = "CK\\CK+48"
    train_images = np.empty([1,48*48])
    test_images= np.empty([1, 48*48])
    train_labels,test_labels=[],[]
    folder = [f for f in os.listdir(path)]
    label = 0

    for f in folder :
        print(f)

        image_list = [im for im in os.listdir(path + "\\" + f) if im.endswith('.png')]
        pack = []

        for i in image_list:
            img1 = cv2.imread(path + "\\" + f + "\\" + i)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img1 = img1 / 255.0
            img1 = np.reshape(img1, 48*48)
            pack = np.concatenate((pack, img1), axis=0)

        pack = pack.reshape(len(image_list), 48*48)
        train = pack[0:int(np.floor(len(pack)*0.8)), :]
        test  = pack[int(np.ceil(len(pack)*0.8)):, :]
        train = train.reshape(len(train), 48*48)

        train_images = np.concatenate((train_images, train), axis = 0)
        test_images  = np.concatenate((test_images, test), axis = 0)
        train_labels = np.concatenate((train_labels, np.ones(len(train))*label))
        test_labels  = np.concatenate((test_labels, np.ones(len(test))*label))
        label = label + 1

    train_images = train_images[1:, :]
    test_images  = test_images[1:, :]


    train_images, train_labels = shuffle(train_images, train_labels)
    test_images, test_labels = shuffle(test_images, test_labels)
    print(train_labels.shape)
    return train_images, train_labels, test_images, test_labels

def data_prep_fer():
    df=pd.read_csv('fer2013\\fer2013.csv')
    train_images,train_labels,test_images,test_labels=[],[],[],[]

    for index, row in df.iterrows():
        val=row['pixels'].split(" ")
        if 'Training' in row['Usage']:
           train_images.append(np.array(val,'float32'))
           train_labels.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           test_images.append(np.array(val,'float32'))
           test_labels.append(row['emotion'])

    train_images = np.array(train_images,'float32')
    # train_images -= np.mean(train_images, axis=0)
    # train_images /= np.std(train_images, axis=0)


    train_images = train_images/255
    test_images  = np.array(test_images,'float32')
    # test_images -= np.mean(test_images, axis=0)
    # test_images /= np.std(test_images, axis=0)
    test_images  = test_images/255

    train_images, train_labels = shuffle(train_images, train_labels)
    test_images, test_labels = shuffle(test_images, test_labels)

    return train_images, train_labels, test_images, test_labels


def create_model_mlp():
    model = keras.Sequential([
        # keras.layers.Flatten(input_shape=(48, 48)),  #convert 28*28 matrix to 1 vector len(28*28)
        keras.layers.Dense(256, activation='relu'), # 128 FC
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

    return model

def create_model_cnn(train_images, num_labels):
    model = keras.Sequential()

    print(train_images.shape[1:])

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(train_images.shape[1:])))
    # model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    # model.add(Dropout(0.2))

    #2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    # model.add(Dropout(0.2))

    # #3rd convolution layer
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

    model.add(Flatten())

    #fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_labels, activation='softmax'))

    return model


def train_cnn():
    train_images, train_labels, test_images, test_labels = data_prep_CK1()
    train_labels =  keras.utils.to_categorical(train_labels, 7)
    test_labels  =  keras.utils.to_categorical(test_labels, 7)
    # img = train_images[7].reshape(48, 48)
    # img = img/255.0

    train_images = train_images.reshape(train_images.shape[0], 48,48, 1)
    test_images = test_images.reshape(test_images.shape[0], 48, 48, 1)
    model = create_model_cnn(train_images, 7)
    model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=35, batch_size=10, validation_split=0.2)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")
    print('\nTest accuracy:', test_acc)

    return 1




#
# train_images, train_labels, test_images, test_labels = data_prep_CK1()
# train_cnn()
