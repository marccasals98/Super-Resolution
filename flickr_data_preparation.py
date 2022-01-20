"""

The main goal of this project is to manage the data of the Flickr dataset.

We have to main problems with the images in this dataset:

1 - The image is not displayed in two folders to train and test.

2 - There is a correlation in the order of the images and what appear in them: When we have 'blocks' of image of a
    certain type, for example we have a block of images of American Football all together.

"""
import os
from random import shuffle
from glob import glob
import numpy as np
import cv2
from PIL import Image

files = glob('D:/flickr30k_images/flickr30k_images/*.jpg')
shuffle(files)

# Create the train folder
train_dir = 'D:/flickr30k_images/train'
os.mkdir(train_dir)

# Create the validation folder
validation_dir = 'D:/flickr30k_images/validation'
os.mkdir(validation_dir)

# Create the test folder
test_dir = 'D:/flickr30k_images/test'
os.mkdir(test_dir)

original_dataset_dir = 'D:/flickr30k_images/flickr30k_images'

total_data = []


def create_total_data():
    for img in os.listdir(original_dataset_dir):
        try:
            img_array = cv2.imread(os.path.join(original_dataset_dir, img))
            total_data.append([img_array])
        except Exception as e:
            print('There has been an error ')


create_total_data()
shuffle(total_data)

train_data = total_data[0:25427]
validation_data = total_data[25427:28605]
test_data = total_data[28605:31783]


def copy_images(list, dir):
    for i in range(len(list)):
        try:
            image = list[i]
            image = np.array(image)
            image = image.reshape(np.shape(list[i])[1], np.shape(list[i])[2], 3)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(np.asarray(image))
            image.save(dir + '/' + str(i) + '.jpg')
        except Exception:
            print('Error')


copy_images(train_data, train_dir)
copy_images(validation_data, validation_dir)
copy_images(test_data, test_dir)
