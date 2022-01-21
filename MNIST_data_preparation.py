# This is a first example. The goal is to solve the superresolution problem using MNIST dataset
# First of all we should divide the MNIST dataset into train images and test images
from tensorflow.keras.datasets import mnist
from PIL import Image
import os

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Find a way to only import the images?

# To re-arrange the code pulse Ctrl + Alt + L


# Create MNIST folder
base_dir = 'D:/MNIST'  # naming the path
os.mkdir(base_dir)  # creating the folder

# Create the folder for training
train_dir = 'D:/MNIST/training'
os.mkdir(train_dir)

# Create the folder for testing
test_dir = 'D:/MNIST/testing'
os.mkdir(test_dir)

# directory to store the dataset with smaller images
small_train = 'D:/MNIST/training/small_train'
os.mkdir(small_train)

# directory to store the dataset with normal images
normal_train = 'D:/MNIST/training/normal_train'
os.mkdir(normal_train)

# directory to store the dataset of testing smaller images
small_test = 'D:/MNIST/testing/small_test'
os.mkdir(small_test)

# directory to store the dataset of testing normal images
normal_test = 'D:/MNIST/testing/normal_test'
os.mkdir(normal_test)

# We create a loop for each TRAINING NORMAL image:
for i in range(len(train_images)):
    array_image = train_images[i]
    real_image = Image.fromarray(array_image)
    # real_image.save(globals()['D:/MNIST/training/normal_train/normal_train_image%s.png' % i])
    real_image.save('D:/MNIST/training/normal_train/normal_train_image' + str(i) + '.png')

# Same loop for testing
for i in range(len(test_images)):
    array_image = test_images[i]
    real_image = Image.fromarray(array_image)
    real_image.save('D:/MNIST/testing/normal_test/normal_test_image' + str(i) + '.png')

# Now we can do the same but with the resize images
newImage_size = 14
for i in range(len(train_images)):
    array_image = train_images[i]
    real_image = Image.fromarray(array_image)
    real_image = real_image.resize((newImage_size, newImage_size))  # Resizing the original image to a 14x14 one.
    real_image.save('D:/MNIST/training/small_train/small_train_image' + str(i) + '.png')

for i in range(len(test_images)):
    array_image = test_images[i]
    real_image = Image.fromarray(array_image)
    real_image = real_image.resize((newImage_size, newImage_size))
    real_image.save('D:/MNIST/testing/small_test/small_test_image' + str(i) + '.png')
