import cv2
import numpy as np
import tensorflow as tf

new_model = tf.keras.models.load_model('my_model3')

img = cv2.imread( "test_data/mask_20.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to RGB
img = cv2.resize(img , (224, 224))
img = np.expand_dims(img , axis = 0 )
img = img/255

p = new_model.predict(img)

mask =p[0][0]
no_mask = p[0][1]
if mask > no_mask :
    print ( "mask")
else :
    print ( "no_mask")
