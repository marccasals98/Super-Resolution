# IMPORT LIBARIES AND MODEL
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import PIL.ImageOps
from google.colab import drive

INICIAL_DIMENSION = 64
FINAL_DIMENSION = 256


flickr_model = load_model('name_of_the_model')

#IMPORT THE IMAGE WE WANT TO TRAIN AND PREDICT THE MODEL.
data_image = cv2.imread("image_path")
data_image = data_image.astype('float32')/255
data_image = cv2.resize(data_image, dsize=(INICIAL_DIMENSION, INICIAL_DIMENSION))

data_image_array = np.array(data_image)

# /content/drive/MyDrive/TFG/FOTOS/butterfly_GT.bmp posar aixo per fer la papallona

data_image_array = data_image_array.reshape(1, INICIAL_DIMENSION, INICIAL_DIMENSION, 3)

newImage = flickr_model.predict(data_image_array)
newImage = newImage.reshape(FINAL_DIMENSION, FINAL_DIMENSION, 3)
newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)

data_image_array = data_image_array.reshape(INICIAL_DIMENSION,INICIAL_DIMENSION,3)
data_image_array = cv2.cvtColor(data_image_array, cv2.COLOR_BGR2RGB)


fig, ax = plt.subplots(1,2)
fig.set_size_inches(8,6)
ax[0].imshow((data_image_array*255).astype(np.uint8), cmap=plt.cm.binary)
ax[1].imshow((newImage*255).astype(np.uint8), cmap=plt.cm.binary)
plt.show()