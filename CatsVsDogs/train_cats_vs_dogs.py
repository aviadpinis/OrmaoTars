# coding: utf-8

import keras
from keras.models import Sequential
from keras.layers.core  import Dense
from keras.optimizers  import Adam

from keras.preprocessing.image  import ImageDataGenerator

train_path = './data/generate/training/'
validation_path = './data/generate/validation/'

IMAGE_SIZE = 224

train_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(train_path, target_size=(IMAGE_SIZE,IMAGE_SIZE), classes=['cat','dog'], batch_size=100)
validation_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_path, target_size=(IMAGE_SIZE,IMAGE_SIZE), classes=['cat','dog'], batch_size=20)

vgg16_model = keras.applications.vgg16.VGG16()

model = Sequential()
for idx,layer in enumerate(vgg16_model.layers):
    if idx < len(vgg16_model.layers)-1:
        model.add(layer)    

for layer in model.layers:
    layer.trainable = False

model.add(Dense(2,activation='softmax'))

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches,
                    steps_per_epoch=18,
                    validation_data=validation_batches,
                    validation_steps=10,
                    epochs=40,
                    verbose=2)

model.save("model.h5")