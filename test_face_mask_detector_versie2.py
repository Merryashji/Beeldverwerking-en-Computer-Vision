from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

prototxtPath = r'C:\Users\Gebruiker\Downloads\caffe_model_for_dace_detection-master\deploy.prototxt'
weightsPath = r'C:\Users\Gebruiker\Downloads\caffe_model_for_dace_detection-master\res10_300x300_ssd_iter_140000.caffemodel'
model = load_model(r'C:\Users\Gebruiker\Downloads\my_model3')
image = cv2.imread(r'C:\Users\Gebruiker\Downloads\test_data\mask_17.png')
face_net = cv2.dnn.readNet(prototxtPath, weightsPath)


def test_face_mask(image, face_net, model):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 244), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locations = []
    predicts = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locations.append((startX, startY, endX, endY))

    faces = np.array(faces, dtype="float32")
    predicts = model.predict(faces)
    return (locations, predicts)


locations, predicts = test_face_mask(image, face_net, model)
for box, pred in zip(locations, predicts):
    (startX, startY, endX, endY) = box
    mask, no_mask = pred
    if mask > no_mask:
        print("mask")
    else:
        print("no_mask")
    label = 'Mask' if mask > no_mask else 'No Mask'
    color = (255, 255, 255) if label == 'Mask' else (0, 0, 0)
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
cv2.imshow("test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
