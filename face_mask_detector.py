import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths
import cv2

from tensorflow.keras.preprocessing.image import img_to_array


image_paths=list(paths.list_images('C:/Users/Gebruiker/Downloads/data'))
training_data =[]
labels=[]

def create_training_data():
    for i in image_paths:
        label=i.split(os.path.sep)[-2]
        labels.append(label)
        img = cv2.imread(i)
        img_size = 224
        img =cv2.resize(img  , (img_size, img_size))
        img = img_to_array(img )
        training_data.append(img )

create_training_data()
