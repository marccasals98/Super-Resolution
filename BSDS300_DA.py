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

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=2500)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

# Path to the data:
train_path = "D:/BSDS300/images/train"
test_path = "D:/BSDS300/images/test"

# IMAGE GENERATOR:
if os.path.exists('D:/BSDS300/images/train/2092_0.jpg') is False:
    gen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.15,
                             zoom_range=0.1,
                             channel_shift_range=10.,
                             horizontal_flip=True)

    for img in os.listdir(train_path):
        basename = os.path.basename(os.path.realpath(img))  # We resize the image but with the same name.
        image_path = 'D:/BSDS300/images/train/' + str(basename)
        basename = basename.replace('.jpg','')
        assert os.path.isfile(image_path)
        image = cv2.resize(cv2.imread(image_path), dsize=(300, 300))
        image = np.expand_dims(image, 0)
        aug_iter = gen.flow(image)  # Generate batches of augmented images from this image
        aug_images = [next(aug_iter)[0].astype(np.uint8).astype('float32') / 255
                      for i in range(10)]  # List of augmented images
        for i in range(10):
            aug_images[i] = (aug_images[i] * 255).astype(np.uint8)
            aug_images[i] = cv2.cvtColor(aug_images[i], cv2.COLOR_BGR2RGB)
            aug_images[i] = Image.fromarray(aug_images[i])
            aug_images[i].save('D:/BSDS300/images/train/' + str(basename)+'_'+str(i)+'.jpg')


def plotImage(image_array):
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    plt.imshow((image_array * 255).astype(np.uint8))
    plt.show()


# Tensorboard declaration
NAME = 'BSDS300-{}'.format(int(time.time()))  # Name of our try
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

'''
IMAGE READING 

We need to create a custom data generator that will read our images in batches. That's because the default data
generator works poorly in the problems that are not classification. Using this, we have more control to what we 
want to achieve.
 
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
            image = cv2.resize(image, (300, 300))  # Original size image
            images.append(image)
        images_resized = []
        for id in batch_IDs:
            path = os.path.join(self.directory, id)
            image = cv2.imread(path)
            image = image.astype('float32') / 255
            image = cv2.resize(image, (150, 150))  # Resized image
            images_resized.append(image)
        # dataset = [images_resized,images]
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
m = 4
upscaling = 2
activation = tf.keras.layers.PReLU(shared_axes=[1,2])

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
          epochs=500,
          batch_size=30,  # 30
          # validation_steps=4,
          # steps_per_epoch=20,
          # validation_batch_size=10,  # 20
          # validation_split=0.2,
          validation_data=test_generator,
          callbacks=[tensorboard])

print(model.evaluate(test_generator))

# SAVING THE MODEL
if os.path.isfile('BSDS300.h5') is False:
    model.save('BSDS300.h5')
# DEMO-----------


dir = 'C:/Users/marcc/OneDrive/Escritorio'
os.chdir(dir)

myImage = Image.open("3096.jpg")  # convert to black and white
# myImage = PIL.ImageOps.invert(myImage)
myImage = myImage.resize((150, 150))
plt.imshow(myImage)
plt.show()

myImage = cv2.imread("C:/Users/marcc/OneDrive/Escritorio/3096.jpg")
myImage = myImage.astype('float32') / 255
myImage = cv2.resize(myImage, dsize=(150, 150))

myImage_array = np.array(myImage)

plt.imshow(myImage_array, cmap=plt.cm.binary)
plt.show()

myImage_array = myImage_array.reshape(1, 150, 150, 3)

newImage = model.predict(myImage_array)
newImage = newImage.reshape(300, 300, 3)
# newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)
# plt.imshow((newImage * 255).astype(np.uint8))
# plt.show()
plotImage(newImage)



#tensorboard --logdir=C:\Users\marcc\PycharmProjects\up_scaling_IA_2.0\logs
