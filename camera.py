#!/usr/bin/python3

from tensorflow.keras import models
from PIL import Image, ImageDraw
import numpy as np
import cv2

from common import IM_SIZE, N_LABELS, LABELS


def face_expression(model):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img_raw = cam.read()
        img_raw_bw = np.dot(img_raw[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        img = Image.fromarray(img_raw_bw, 'L')
        img_resized = img.resize((48, 48), Image.ANTIALIAS)
        X = (np.array(img_resized) / 255.).reshape(1, *IM_SIZE, 1)
        y_hat = model.predict(X)
        description = LABELS[y_hat.argmax()] + ' ' + str(y_hat.max()*100) + '%'

        # print(description)

        img_colored = Image.fromarray(img_raw);
        ImageDraw.Draw(img_colored).text((0, 0), description, (255, 255, 255))

        cv2.imshow('Face expression', np.array(img_colored).astype(np.uint8))
        if cv2.waitKey(1) == 27: 
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = models.load_model('model.h5')
    face_expression(model)
