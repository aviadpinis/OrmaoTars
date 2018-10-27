# coding: utf-8

from keras.models import Sequential
from keras.preprocessing.image  import ImageDataGenerator

test_path = './test/'
IMAGE_SIZE = 224

test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(IMAGE_SIZE,IMAGE_SIZE), classes=['cat','dog'], batch_size=20)

model = Sequential()
model.load("model.h5")

model.predict_generator(test_batches,steps=1, verbose=0)