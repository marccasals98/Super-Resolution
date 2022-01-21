'''

SUPERRESOLUTION using BSDS300 dataset

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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

def plotImage(image_array):
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    plt.imshow((image_array * 255).astype(np.uint8))
    plt.show()


'''
CREATING THE DATASETS:
 
First of all we need to create the datasets, for doing so we downloaded an existing dataset of images and now we have
to resize them in order to make the learning possible.

'''
if os.path.exists('D:/BSDS300/images/train_small') is False:  # This conditional checks if the folders have been created
    print('Making the folders...')
    # Create the folder where we will put the training downscaled images
    train_small_dir = 'D:/BSDS300/images/train_small'
    os.mkdir(train_small_dir)

    # Create the folder where we will put the testing downscaled images
    test_small_dir = 'D:/BSDS300/images/test_small'
    os.mkdir(test_small_dir)

    # Directions of the other datasets:
    train_normal_dir = 'D:/BSDS300/images/train'
    test_normal_dir = 'D:/BSDS300/images/test'

    # Transform real_size training images to small images.
    for img in os.listdir(train_normal_dir):
        try:
            image = cv2.imread(os.path.join(train_normal_dir, img))
            basename = os.path.basename(os.path.realpath(img))  # We resize the image but with the same name.
        except Exception as e:
            print("Problem reading an image in train directory")

        resized_image = cv2.resize(image, dsize=(image.shape[1] // 2, image.shape[0] // 2))
        if resized_image is None:
            print("print if it's None")
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)  # We need this bc PIL uses BGR instead of RGB
        resized_image = Image.fromarray(resized_image)
        resized_image.save('D:/BSDS300/images/train_small/' + str(basename))

    # Transform real_size test images to small images.
    for img in os.listdir(test_normal_dir):
        try:
            image = cv2.imread(os.path.join(test_normal_dir, img))
            basename = os.path.basename(os.path.realpath(img))  # We resize the image but with the same name.
        except Exception as e:
            print("Problem reading an image in test directory")

        resized_image = cv2.resize(image, dsize=(image.shape[1] // 2, image.shape[0] // 2))
        if resized_image is None:
            print("print if it's None")
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)  # We need this bc PIL uses BGR instead of RGB
        resized_image = Image.fromarray(resized_image)
        resized_image.save('D:/BSDS300/images/test_small/' + str(basename))

'''
CREATING THE NETWORK:

In this step we are creating the network and training it with the images of the dataset

'''

# Tensorboard declaration
NAME = 'BSDS300_GRAPHPLOTTER-{}'.format(int(time.time()))  # Name of our try
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# Path to the data:
train_small_path = "D:/BSDS300/images/train_small"
train_normal_path = "D:/BSDS300/images/train"

test_small_path = "D:/BSDS300/images/test_small"
test_normal_path = "D:/BSDS300/images/test"

''' Image reading from the directories. I know we already read the images but we want to execute the first part
 of the code only once.'''

train_small_array = []
for img in os.listdir(train_small_path):
    try:
        image = cv2.imread(os.path.join(train_small_path, img))
        image = image.astype('float32') / 255
        image = cv2.resize(image, dsize=(150, 150))
        train_small_array.append(image)
    except Exception as e:
        print("problem with image reading in train small")
        pass
train_normal_array = []
for img in os.listdir(train_normal_path):
    try:
        image = cv2.imread(os.path.join(train_normal_path, img))
        image = image.astype('float32') / 255
        image = cv2.resize(image, dsize=(300, 300))
        train_normal_array.append(image)
    except Exception as e:
        print("problem with image reading in train normal")
        pass
test_small_array = []
for img in os.listdir(test_small_path):
    try:
        image = cv2.imread(os.path.join(test_small_path, img))
        image = image.astype('float32') / 255
        image = cv2.resize(image, dsize=(150, 150))
        test_small_array.append(image)
    except Exception as e:
        print("problem with image reading in test small")
        pass

test_normal_array = []
for img in os.listdir(test_normal_path):
    try:
        image = cv2.imread(os.path.join(test_normal_path, img))
        image = image.astype('float32') / 255
        image = cv2.resize(image, dsize=(300, 300))
        test_normal_array.append(image)
    except Exception as e:
        print("problem with image reading in test normal")
        pass

'''
DATA AUGMENTATION
 
Creating the datagen that will allow us to make more images out of the ones we already have:
'''
# gen = ImageDataGenerator(rotation_range=10,
#                          width_shift_range=0.1,
#                          height_shift_range=0.1,
#                          shear_range=0.15,
#                          zoom_range=0.1,
#                          channel_shift_range=10.,
#                          horizontal_flip=True)
#
# # Image generator train_small
# for img in os.listdir(train_small_path):
#     basename = os.path.basename(os.path.realpath(img))  # We resize the image but with the same name.
#     image_path = 'D:/BSDS300/images/train_small/' + str(basename)
#     assert os.path.isfile(image_path)
#     image = cv2.resize(cv2.imread(image_path), dsize=(150, 150))
#     image = np.expand_dims(image, 0)
#     aug_iter = gen.flow(image)  # Generate batches of augmented images from this image
#     aug_images = [next(aug_iter)[0].astype(np.uint8).astype('float32') / 255
#                   for i in range(10)]  # List of augmented images
#     train_small_array = train_small_array + aug_images
#
# # Image generator train_normal
# for img in os.listdir(train_normal_path):
#     basename = os.path.basename(os.path.realpath(img))  # We resize the image but with the same name.
#     image_path = 'D:/BSDS300/images/train/' + str(basename)
#     assert os.path.isfile(image_path)
#     image = cv2.resize(cv2.imread(image_path), dsize=(300, 300))
#     image = np.expand_dims(image, 0)
#     aug_iter = gen.flow(image)  # Generate batches of augmented images from this image
#     aug_images = [next(aug_iter)[0].astype(np.uint8).astype('float32') / 255
#                   for i in range(10)]  # List of augmented images
#     train_normal_array = train_normal_array + aug_images

# Reshaping to feed the network.
train_small_array = np.array(train_small_array).reshape((200, 150, 150, 3))
train_normal_array = np.array(train_normal_array).reshape((200, 300, 300, 3))

test_small_array = np.array(test_small_array).reshape((100, 150, 150, 3))
test_normal_array = np.array(test_normal_array).reshape((100, 300, 300, 3))

# Making the whole set:
training_data = [train_small_array, train_normal_array]
testing_data = [test_small_array, test_normal_array]

# MAKING THE NETWORK
d = 56
s = 12
m = 4
upscaling = 2

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
model.compile(loss='mse',
              optimizer='adam',  # 'adam' optimizers.RMSprop(learning_rate=1e-4)
              metrics=['acc'])

# Fast version:
model.fit(x=train_small_array, y=train_normal_array,
          epochs=500,
          batch_size=30,  # 30
          # validation_steps=4,
          # steps_per_epoch=20,
          validation_batch_size=10,  # 20
          validation_split=0.2,
          callbacks=[tensorboard])

print(model.evaluate(test_small_array, test_normal_array))
if os.path.isfile('BSDS300_primitive.h5') is False:
    model.save('BSDS300_primitive.h5')
# DEMO-----------


dir = 'C:/Users/marcc/OneDrive/Escritorio'
os.chdir(dir)

myImage = Image.open("captura.jpeg")  # convert to black and white
# myImage = PIL.ImageOps.invert(myImage)
myImage = myImage.resize((150, 150))
plt.imshow(myImage)
plt.show()

myImage = cv2.imread("C:/Users/marcc/OneDrive/Escritorio/captura.jpeg")
myImage = myImage.astype('float32') / 255
myImage = cv2.resize(myImage, dsize=(150, 150))

myImage_array = np.array(myImage)

plt.imshow(myImage_array, cmap=plt.cm.binary)
plt.show()

myImage_array = myImage_array.reshape(1, 150, 150, 3)

newImage = model.predict(myImage_array)
newImage = newImage.reshape(300, 300, 3)
newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)
# plt.imshow((newImage * 255).astype(np.uint8))
# plt.show()
plotImage(newImage)
