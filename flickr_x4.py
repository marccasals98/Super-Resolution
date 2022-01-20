'''

SUPERRESOLUTION upscaling x4 using Flickr Dataset

Marc Casals 2021

'''

from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import initializers
import time
from random import shuffle

INPUT_DIM = 64
OUTPUT_DIM = 256

# Some definitions for creating and shuffling the dataset.
train_dir = 'D:/flickr30k_images/train'
test_dir = 'D:/flickr30k_images/test'
original_dataset_dir = 'D:/flickr30k_images/flickr30k_images'


def create_total_data():
    for img in os.listdir(original_dataset_dir):
        try:
            img_array = cv2.imread(os.path.join(original_dataset_dir, img))
            total_data.append([img_array])
        except Exception as e:
            print('There has been an error :(')

def copy_images(list, dir):
    i = 0
    for i in range(len(list)):
        try:
            image = list[i]
            image = np.array(image)
            image = image.reshape(np.shape(list[i])[1], np.shape(list[i])[2], 3)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(np.asarray(image))
            image.save(dir + '/' + str(i) + '.jpg')
        except Exception as e:
            print('Error')


if os.path.exists(train_dir) is False:
    print('Creating and Shuffling Flickr Dataset')
    total_data = []
    create_total_data()
    shuffle(total_data)

    train_data = total_data[0:25427]
    test_data = total_data[25427:31783]

    copy_images(train_data, train_dir)
    copy_images(test_data, test_dir)

# Path to the data:
train_path = "D:/flickr30k_images/train"
test_path = "D:/flickr30k_images/test"


def plotImage(image_array):
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    plt.imshow((image_array * 255).astype(np.uint8))
    plt.show()


# Tensorboard declaration
NAME = 'flickr_test1{}'.format(int(time.time()))  # Name of our try
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

'''
IMAGE READING 

We need to create a custom data generator that will read our images in batches. That's because the default data
generator works poorly in the problems that are not classification. Using this, we have more control to what we 
want to achieve. 

It's important to remark that in this example we don't have our data split in train and test folders, so we will have 
to split it ourselves.

'''


# For creating the data generator, first we need to create the class:
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, dataset_directory):  # batch_size and dataset_directory are attributes of the class
        self.batch_size = batch_size  # We need to reassign the attributes from those we passed.
        self.directory = dataset_directory
        self.list_IDs = os.listdir(self.directory)

        # Returns the number of batches to generate

    def __len__(self):
        return len(self.list_IDs) // self.batch_size

        # Return a batch of given index
        # Create your logic how you want to load your data

    def __getitem__(self, index):  # This method generates a batch of data
        batch_IDs = self.list_IDs[index * self.batch_size: (index + 1) * self.batch_size]
        images = []
        for id in batch_IDs:
            path = os.path.join(self.directory, id)
            image = cv2.imread(path)
            image = image.astype('float32') / 255
            image = cv2.resize(image, (OUTPUT_DIM, OUTPUT_DIM))  # Original size image
            images.append(image)
        images_resized = []
        for id in batch_IDs:
            path = os.path.join(self.directory, id)
            image = cv2.imread(path)
            image = image.astype('float32') / 255
            image = cv2.resize(image, (INPUT_DIM, INPUT_DIM))  # Resized image
            images_resized.append(image)
        images = np.asarray(images)  # IMPORTANT: the output must be np array, otherwise we can't train
        images_resized = np.asarray(images_resized)
        return images_resized, images  # We return both so we can train as normally.


train_generator = CustomDataGenerator(
    batch_size=30,
    dataset_directory=train_path
)

test_generator = CustomDataGenerator(
    batch_size=30,
    dataset_directory=test_path
)

# MAKING THE NETWORK
d = 56
s = 12
m = 7
upscaling = 4
activation = tf.keras.layers.PReLU(shared_axes=[1, 2])

model = models.Sequential()
bias = True

# Feature extraction:
model.add(layers.Conv2D(filters=d,
                        kernel_size=5,
                        padding='SAME',
                        data_format="channels_last",
                        use_bias=bias,
                        kernel_initializer=initializers.he_normal(),
                        input_shape=(None, None, 3),
                        activation='relu'))

# Shrinking:
model.add(layers.Conv2D(filters=s,
                        kernel_size=1,
                        padding='same',
                        use_bias=bias,
                        kernel_initializer=initializers.he_normal(),
                        activation='relu'))

for i in range(m):
    model.add(layers.Conv2D(filters=s,
                            kernel_size=3,
                            padding="same",
                            use_bias=bias,
                            kernel_initializer=initializers.he_normal(),
                            activation='relu'),

              )

# Expanding
model.add(layers.Conv2D(filters=d,
                        kernel_size=1,
                        padding='same',
                        use_bias=bias,
                        kernel_initializer=initializers.he_normal,
                        activation='relu'))

# Deconvolution
model.add(layers.Conv2DTranspose(filters=3,  # 1   #3
                                 kernel_size=9,
                                 strides=(upscaling, upscaling),
                                 padding='same',
                                 use_bias=bias,
                                 kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.001),
                                 activation='relu'))

# MODEL COMPILATION
model.compile(loss='mse',  # mse
              optimizer='adam',  # 'adam' optimizers.RMSprop(learning_rate=1e-4)
              metrics=['acc'])

# Fast version:
model.fit(train_generator,
          epochs=50,
          batch_size=30,  # 30
          # validation_steps=4,
          # steps_per_epoch=20,
          # validation_batch_size=10,  # 20
          validation_data=test_generator,
          # validation_split=0.2,
          callbacks=[tensorboard])

print(model.evaluate(test_generator))

# IMPROVISED TUTORIAL: HOW TO USE TENSORBOARD
# Execute cmd from the tensorboard.exe folder
# Write the following code: tensorboard --logdir=C:\Users\marcc\PycharmProjects\flickr_x4\logs
# The directory is where the folder logs is created
# It will reply with a localhost link that you will have to copy it to your browser.
# In my case: http://localhost:6006/

# SAVING THE MODEL
if os.path.isfile('flickrx4SR.h5') is False:
    model.save('flickrx4SR.h5')

# DEMO-----------
myImage = cv2.imread("C:/Users/marcc/OneDrive/Escritorio/captura.jpeg")
myImage = myImage.astype('float32') / 255
myImage = cv2.resize(myImage, dsize=(INPUT_DIM, INPUT_DIM))

myImage_array = np.array(myImage)

plt.imshow(myImage_array, cmap=plt.cm.binary)
plt.show()

myImage_array = myImage_array.reshape(1, INPUT_DIM, INPUT_DIM, 3)

newImage = model.predict(myImage_array)
newImage = newImage.reshape(OUTPUT_DIM, OUTPUT_DIM, 3)
plotImage(newImage)
