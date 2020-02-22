
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import keras

from facial_expression import data_prep_CK1
# match = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
# match = {0: 'happy', 1:'sad', 2:'surprise' }
match = ('angry', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

loaded_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
loaded_model.summary()

def expression(img):
    img = img/255
    img = np.resize(img, (48,48))
    img = img.reshape(48,48, 1)
    predictions = loaded_model.predict(np.array([img]))
    print(predictions)
    return np.argmax(predictions[0])

def replace_face(img, x, y, w, h, statut = 2):
    name = match[statut]
    print(name)
    emoji = cv2.imread('data//' + name + '.png')
    emoji = cv2.resize(emoji, (w, h), interpolation = cv2.INTER_AREA)
    img2gray = cv2.cvtColor(emoji,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    roi = img[y:y+h, x:x+w]
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    img2_fg = cv2.bitwise_and(emoji,emoji,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)
    img[y:y+h, x:x+w] = dst

    return img

def frame_face_with_haar(img, scaleFactor=1.1, minNeigh=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classif = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    faces = face_classif.detectMultiScale(gray, scaleFactor, minNeigh)
    for (x, y ,w, h) in faces:
        # cv2.rectangle(img, (x, y) , (x+w, y+h), (0, 255, 0), 2 )
        img=replace_face(img, x, y, w, h, expression(gray[y:y+h, x:x+w]))
    return img

def frame_face_with_mcnn(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = MTCNN()
    results = detector.detect_faces(img)
    if results:
        x, y, w, h = results[0]['box']
        img = replace_face(img, x, y ,w ,h, expression(gray[y:y+h, x:x+w]))
        #cv2.rectangle(img, (x, y) , (x+w, y+h), (0, 255, 0), 2 )

    return img


def test_webcam():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = frame_face_with_haar(frame)
        # Display the resulting frame
        cv2.imshow('frame',gray)
        # time.sleep(0.2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return 1

def test_on_dataset():
    train_images, train_labels, test_images, test_labels = data_prep_CK1()
    print(train_labels)
    for i in range(10):
        img = train_images[i]
        plt.imshow(img.reshape(48,48))
        # print(match[train_labels[i]])
        img = img/255.0
        img = np.resize(img, (48,48))
        img = img.reshape(48,48, 1)

        predictions = loaded_model.predict(np.array([img]))
        print(predictions)
        print(match[np.argmax(predictions[0])])
        plt.show()
        cv2.waitKey(0)

test_webcam()
