import numpy as np
import os
import matplotlib.pyplot as plt
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



#Data preprocessing
image_paths = r"C:\Users\mimii\Downloads\data"
dataset  = ["with_mask", "without_mask"]

training_data =[]
labels=[]

def data_preprocessing():
    for i in dataset :
        path = os.path.join(image_paths, i)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            training_data.append(image)
            labels.append(i)
            
            
data_preprocessing()

training_data=np.array(training_data,dtype='float32') 
labels=np.array(labels)
lb=LabelBinarizer() 
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

#split data into test data en train data 
train_X,test_X,train_Y,test_Y=train_test_split(training_data,labels,test_size=0.30,stratify=labels,random_state=20)
print (len(train_X))
print (len(test_Y))

#ImageDataGenerator to expand the dataset. 
data=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,vertical_flip=True,fill_mode='nearest')

#building neural network using mobileNetV2 and 5 layers 
baseModel=MobileNetV2(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))
baseModel.summary()

headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name='Flatten')(headModel)
headModel=Dense(64,activation='relu')(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation='softmax')(headModel)
model=Model(inputs=baseModel.input,outputs=headModel)
model.summary()   
      
for layer in baseModel.layers:
    layer.trainable=False
learning_rate= 0.001
Epochs= 20
Batch_size= 32
print (len(train_X)//Batch_size)
optimizer=Adam(lr=learning_rate,decay=learning_rate/Epochs)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
H = model.fit(data.flow(train_X,train_Y,Batch_size), steps_per_epoch=len(train_X)//Batch_size, validation_data=(test_X,test_Y), validation_steps=len(test_X)//Batch_size, epochs=Epochs)
model.save("face_mask_detector_model" , save_format = "h5")

#evaluate the model
test_loss, test_acc = model.evaluate(test_X,  test_Y, verbose=2)

# plotting accuracy and validation accuracy
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title('accuracy graph')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'])
plt.show()



# plotting loss and validation loss
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('loss graph')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'])
plt.show()
