#!/usr/bin/python3

from tensorflow.keras import models
from PIL import Image, ImageDraw
import numpy as np
import cv2

from common import MODELS_PATH, IM_SIZE, LABELS


def get_faces(image_gray, face_cascade):
    ''' Detect faces in the photo '''
    faces = face_cascade.detectMultiScale(image_gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    return faces

def face_expression(model, face_cascade):
    ''' Recognizes human faces and predicts face expressions '''
    cam = cv2.VideoCapture(0)
    while True:
        # Read image from camera
        ret_val, img_raw = cam.read()

        img_raw_bw = np.dot(img_raw[..., :3],
                            [0.299, 0.587, 0.114]).astype(np.uint8)
        img_colored = Image.fromarray(img_raw)

        # Detect faces
        faces = get_faces(img_raw_bw, face_cascade)

        for (x, y, w, h) in faces:
            ImageDraw.Draw(img_colored).rectangle([(x, y), (x + w, y + h)],
                                                  fill=None,
                                                  outline=(0, 255, 0))
            face_arr = img_raw_bw[y:y+h, x:x+w]
            face_arr = Image.fromarray(face_arr, 'L')
            img_resized = face_arr.resize((48, 48), Image.ANTIALIAS)
            X = (np.array(img_resized) / 255.).reshape(1, *IM_SIZE, 1)

            y_hat = model.predict(X)
            predicted_label = LABELS[y_hat.argmax()]
            certainty = int(np.round(y_hat.max() * 100, -1))
            description = predicted_label + ' ' + str(certainty) + '%'
            ImageDraw.Draw(img_colored).text((x+w, y),
                                             description,
                                             (255, 255, 255))
            face_arr = np.array(face_arr).astype(np.uint8)

        open_cv_image = np.array(img_colored).astype(np.uint8)
        cv2.imshow('Face expression', open_cv_image)
        if cv2.waitKey(1) == 27: 
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load face expression recognizer model
    model = models.load_model(MODELS_PATH + 'model.h5')

    # Load Haar-cascade detection model
    casc_path = MODELS_PATH + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(casc_path)

    face_expression(model, face_cascade)
