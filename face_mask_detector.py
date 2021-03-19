import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths
import cv2 

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


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

training_data=np.array(training_data,dtype='float32') 
labels=np.array(labels)

lb=LabelBinarizer() 
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

train_X,test_X,train_Y,test_Y=train_test_split(training_data,labels,test_size=0.30,stratify=labels,random_state=10)
