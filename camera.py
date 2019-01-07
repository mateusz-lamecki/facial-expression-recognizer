#!/usr/bin/python3

from keras import models
import cv2

from common import MODELS_PATH
from game import Game
import utils


if __name__ == '__main__':
    model = models.load_model(MODELS_PATH + 'model.h5')

    casc_path = MODELS_PATH + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(casc_path)

    game = Game(model, face_cascade)
    game.run()
