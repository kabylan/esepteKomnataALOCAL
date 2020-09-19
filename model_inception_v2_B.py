# -*- coding: utf-8 -*-

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#  try:
#    # Currently, memory growth needs to be the same across GPUs
  #   for gpu in gpus:
  #     tf.config.experimental.set_memory_growth(gpu, True)
  #   logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  #   print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  # except RuntimeError as e:
  #   # Memory growth must be set before GPUs have been initialized
  #   print(e)

import numpy as np
import pandas as pd
import imageio
import skimage
import skimage.io
import skimage.transform
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam,RMSprop,SGD
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import matplotlib.pylab as plt
from keras.preprocessing.image import ImageDataGenerator


# Using pre-trained model
# conv_base = InceptionResNetV2(include_top = False, weights = '/home/apollo/Downloads/PretrainedWeights/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape = (200,200,3)) #150,150
# conv_base.trainable = False
# conv_base.summary()


# # build on top of imported model
# model = Sequential()
# model.add(conv_base)
# model.add(Flatten())
# model.add(Dense(512,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(15, activation='softmax'))

# model.compile(Adam(lr=0.0001),loss="categorical_crossentropy", metrics=["accuracy"])
# #model.compile(SGD(lr=0.0001),loss="categorical_crossentropy", metrics=["accuracy"])
# model.summary()


train_data_dir = '/home/apollo/Downloads/IndoorDataset/indoorCVPR_09/ImagesLess'
img_width = 200
img_height = 200
batch_size = 8
nb_epochs = 10
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch = train_generator.samples // batch_size,
#     validation_data = validation_generator, 
#     validation_steps = validation_generator.samples // batch_size,
#     epochs = nb_epochs)

# model.save('inception_v2_200px.h5') 
# model.save_weights('Weightsinception_v2_200px.h5')

from keras.models import load_model

model = load_model('inception_v2_200px.h5')
model.load_weights('Weightsinception_v2_200px.h5')

# # check classification mapping
dict = train_generator.class_indices

# # Graphs
# print(history.history.keys())
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)

# plt.title('Training and validation accuracy')
# plt.plot(epochs, acc, 'red', label='Training acc')
# plt.plot(epochs, val_acc, 'blue', label='Validation acc')
# plt.legend()

# plt.figure()
# plt.title('Training and validation loss')
# plt.plot(epochs, loss, 'red', label='Training loss')
# plt.plot(epochs, val_loss, 'blue', label='Validation loss')

# plt.legend()
# plt.show()

import time
import numpy as np
from keras.preprocessing import image


from os import walk

f = []
for (dirpath, dirnames, filenames) in walk("/home/apollo/Downloads/Stroka.kg"):
    f.extend(filenames)
    break

for file_name in f:

  print(file_name)
  test_image = image.load_img('/home/apollo/Downloads/Stroka.kg/'+file_name, target_size = (200, 200))
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis = 0)
  test_image = preprocess_input(test_image) # added to check same preds issue
  start_time = time.time()
  result = model.predict(test_image)
  #decode_predictions(result)
  print("--- %s seconds ---" % (time.time() - start_time))
  for i in range (0,dict.__len__()):
      if result[0][i] >= 0.5:
          listOfKeys = [key  for (key, value) in dict.items() if value == i]
          for key  in listOfKeys:
              print(key) 
              break
