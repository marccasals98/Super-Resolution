import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
import os
import cv2
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import initializers
import matplotlib.pyplot as plt
import time



# Tensorboard Stuff:
NAME = "MNIST_FSRCNN_GRAPH_R-{}".format(
    int(time.time()))  # This is the name of our try, change it if it's a
# new try.
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))  # defining tensorboard directory.

# Path of the data
train_small_path = "D:/MNIST/training/small_train"
train_normal_path = "D:/MNIST/training/normal_train"

test_small_path = "D:/MNIST/testing/small_test"
test_normal_path = "D:/MNIST/testing/normal_test"

# Image reading from the directories. MNIST is in grayscale so we read it that way.
train_small_array = []
for img in os.listdir(train_small_path):
    try:
        train_small_array.append(np.array(cv2.imread(os.path.join(train_small_path, img), cv2.IMREAD_GRAYSCALE)))
    except Exception as e:
        print("problem with image reading in train small")
        pass
train_normal_array = []
for img in os.listdir(train_normal_path):
    try:
        train_normal_array.append(np.array(cv2.imread(os.path.join(train_normal_path, img), cv2.IMREAD_GRAYSCALE)))
    except Exception as e:
        print("problem with image reading in train normal")
        pass
test_small_array = []
for img in os.listdir(test_small_path):
    try:
        test_small_array.append(cv2.imread(os.path.join(test_small_path, img), cv2.IMREAD_GRAYSCALE))
    except Exception as e:
        print("problem with image reading in test small")
        pass

test_normal_array = []
for img in os.listdir(test_normal_path):
    try:
        test_normal_array.append(cv2.imread(os.path.join(test_normal_path, img), cv2.IMREAD_GRAYSCALE))
    except Exception as e:
        print("problem with image reading in test normal")
        pass



# Reshaping to feed the network.
train_small_array = np.array(train_small_array).reshape((60000, 14, 14, 1))
train_normal_array = np.array(train_normal_array).reshape((60000, 28, 28, 1))

test_small_array = np.array(test_small_array).reshape((10000, 14, 14, 1))
test_normal_array = np.array(test_normal_array).reshape((10000, 28, 28, 1))

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
                        input_shape=(None, None, 1),
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
model.add(layers.Conv2DTranspose(filters=1,
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
          validation_steps=4,
          steps_per_epoch=20,
          validation_batch_size=20,
          validation_split=0.2,
          callbacks=[tensorboard])

# # Old version
# model.fit(x=train_small_array, y=train_normal_array,
#           epochs=20,
#           batch_size=1500,
#           validation_split=0.2,
#           callbacks=[tensorboard])

print(model.evaluate(test_small_array, test_normal_array))

# IMPROVISED TUTORIAL: HOW TO USE TENSORBOARD
# Execute cmd from the tensorboard.exe folder
# Write the following code: tensorboard --logdir=C:\Users\marcc\PycharmProjects\MNIST_superresolution\logs
# The directory is where the folder logs is created
# It will reply with a localhost link that you will have to copy it to your browser.
# In my case: http://localhost:6006/


# SAVING THE MODEL
if os.path.isfile('modelFSRCNN.h5') is False:
    model.save('modelFSRCNN.h5')

print(model.summary())

# -DEMO-----------------------------------------------------------------
from PIL import Image
import os
import PIL.ImageOps

dir = 'C:/Users/marcc/OneDrive/Escritorio'
os.chdir(dir)

myImage = Image.open("Captura.PNG").convert('L')  # convert to black and white
myImage = PIL.ImageOps.invert(myImage)
myImage = myImage.resize((14, 14))
# myImage.show()


myImage_array = np.array(myImage)

plt.imshow(myImage_array, cmap=plt.cm.binary)
plt.show()

# myImage_array = 255-myImage_array
# print(myImage_array.shape)
# myImage_array = myImage_array.reshape((28*28))
#myImage_array = myImage_array.astype('float32') / 255
# myImage_array = np.asarray(myImage_array).flatten()
myImage_array = myImage_array.reshape(1, 14, 14, 1)

# print(myImage_array.shape)
# print(train_images[[1]].shape)

newImage = model.predict(myImage_array)
newImage = newImage.reshape(28, 28)
plt.imshow(newImage, cmap=plt.cm.binary)
plt.show()

myImage_array = myImage_array.reshape(14, 14)

myImage_array = cv2.cvtColor(myImage_array, cv2.COLOR_BGR2RGB)
#newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)


#print(newImage)

fig, ax = plt.subplots(1,2)
fig.set_size_inches(8,6)
ax[0].imshow(255-myImage_array)
ax[1].imshow(newImage, cmap=plt.cm.binary)
plt.show()

# tensorboard --logdir=C:\Users\marcc\PycharmProjects\upscalingIA\logs