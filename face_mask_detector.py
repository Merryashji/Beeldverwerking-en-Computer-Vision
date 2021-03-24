import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
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
        img=preprocess_input(img)
        training_data.append(img )

create_training_data()

training_data=np.array(training_data,dtype='float32') 
labels=np.array(labels)

lb=LabelBinarizer() 
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

train_X,test_X,train_Y,test_Y=train_test_split(training_data,labels,test_size=0.30,stratify=labels,random_state=10)


baseModel=MobileNetV2(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))

headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name='Flatten')(headModel)
headModel=Dense(128,activation='relu')(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation='softmax')(headModel)

model=Model(inputs=baseModel.input,outputs=headModel)

for layer in baseModel.layers:
    layer.trainable=False
    
model.compile(loss = "binary_crossentropy" , optimizer = "adam" , metrics = ["accuracy"] )
model.fit(train_X,train_Y, epochs = 10 , validation_split = 0.1)


model.save("my_model2" , save_format = "h5")


