from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2


prototxtPath = r'C:\Users\mimii\Downloads\caffe_model_for_dace_detection-master\deploy.prototxt'
weightsPath = r'C:\Users\mimii\Downloads\caffe_model_for_dace_detection-master\res10_300x300_ssd_iter_140000.caffemodel'

face_net = cv2.dnn.readNet(prototxtPath, weightsPath)
model = load_model(r'C:\Users\mimii\Downloads\face_mask_detector_model')

photo = input ("choose a photo:\n mask_1.png\n mask_2.png\n mask_3.png\n mask_4.png\n mask_5.png\n mask_6.png\n mask_7.png\n mask_8.png\n mask_9.png\n mask_10.png\n mask_11.png\n mask_12.png\n no_mask_1.png\n no_mask_2.png\n no_mask_3.png\n no_mask_4.png\n no_mask_5.png\n no_mask_6.png\n no_mask_7.png\n no_mask_8.png\n no_mask_9.png\n no_mask_10.png\n no_mask_11.png\n no_mask_12.png\n no_face1.png\n no_face.webp\n")
image = cv2.imread(photo)

#we need height and width
(height, width) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (224, 244), (104.0, 177.0, 123.0))
face_net.setInput(blob)
face_detect = face_net.forward()

confidence = face_detect[0, 0, 0, 2]
if confidence > 0.5:
    rect = face_detect[0, 0, 0, 3:7] * np.array([width, height, width, height])
    (startX, startY, endX, endY) = rect.astype('int')
    face = image[startY:endY, startX:endX]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face=np.expand_dims(face,axis=0)
    p = model.predict(face)
    mask =p[0][0]
    no_mask = p[0][1]
    if mask > no_mask :
        label='Mask'
        color=(255,255,255)
    else :
        label='No Mask'
        color=(0,0,0)
    cv2.putText(image,label,(startX,startY),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
    cv2.rectangle(image,(startX,startY),(endX,endY),color,2)
        
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    

