#!/usr/bin/python
# coding:utf8

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
img = load_img('C:\\Users\\ailab\\PycharmProjects\\\drone\\pic\\image4.jpg')
x = img_to_array(img)
x = x.reshape((1,)+x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='C:\\Users\\ailab\PycharmProjects\\drone\\pic\\4', save_prefix='4', save_format='jpg'):
    i += 1
    if i>60:
        break