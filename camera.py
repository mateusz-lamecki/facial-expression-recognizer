#!/usr/bin/python3

from tensorflow.keras import models
from PIL import Image, ImageDraw
import numpy as np
import cv2

from common import IM_SIZE, N_LABELS, LABELS

cascPath = "haarcascade_frontalface_default.xml"

def get_faces(image):
    faceCascade = cv2.CascadeClassifier(cascPath)
    # Read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print("Found {0} faces!".format(len(faces)))
    return faces


def face_expression(model):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img_raw = cam.read()

        img_raw_bw = np.dot(img_raw[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        # print(description)
        faces = get_faces(img_raw)

        img_colored = Image.fromarray(img_raw);
        for (x, y, w, h) in faces:
            ImageDraw.Draw(img_colored).rectangle([(x, y), (x + w, y + h)], fill=None, outline=(0, 255, 0))
            face_arr = img_raw_bw[x:x+w][y:y+h]
            face_arr = Image.fromarray(face_arr, 'L')
            img_resized = face_arr.resize((48, 48), Image.ANTIALIAS)
            X = (np.array(img_resized) / 255.).reshape(1, *IM_SIZE, 1)
            y_hat = model.predict(X)
            description = LABELS[y_hat.argmax()] + ' ' + str(y_hat.max() * 100) + '%'
            ImageDraw.Draw(img_colored).text((x+w, y), description, (255, 255, 255))

        open_cv_image = np.array(img_colored).astype(np.uint8)
        cv2.imshow('Face expression', open_cv_image)
        if cv2.waitKey(1) == 27: 
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = models.load_model('model.h5')
    face_expression(model)
