# coding: utf-8

import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

path_source = 'data/source/'
path_generating = 'data/generate/'

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def gen(img, prefix, theFormat, path_folder):
    i = 0
    for batch in datagen.flow(img, batch_size=1,
                              save_to_dir=path_generating+path_folder, save_prefix=prefix, save_format=theFormat):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely


dataTypeFolders = os.listdir(path_source)
for dataType in dataTypeFolders:
    for folder in dataType:
        images = os.listdir(path_source+dataType+folder)
        for img_name in images:
            name = ".".join(str(x) for x in img_name.split('.')[:2])
            img = load_img(path_source+dataType+folder+'/'+img_name)
            theFormat = img.format
            img = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            img = img.reshape((1,) + img.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            gen(img,name,theFormat,dataType+folder)

