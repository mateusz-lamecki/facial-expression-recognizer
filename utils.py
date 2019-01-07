import itertools

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    ''' Source: https://scikit-learn.org/stable/auto_examples/ '''

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def draw_text(img, label, position, color, right_side=True, center=False,
              text_size=10):
    ''' Draws text on PIL.Image object '''
    font = ImageFont.truetype('DejaVuSans.ttf', text_size)
    if center:
        position = (position[0]-font.getsize_multiline(label)[0]//2, position[1])
    elif not right_side:
        position = (position[0]-font.getsize_multiline(label)[0], position[1])

    ImageDraw.Draw(img).text(position, label, color, font=font)
    return img


def print_player_status(img, n_faces, score, color, left_player=True):
    score_str = str(int(np.round(score)))
    label = 'Wykryto: ' + str(n_faces) + '\n' + str(score_str) + ' punktów'
    pos = 0 if left_player else img.size[0]
    return draw_text(img, label, (pos, 0), color,
                     right_side=(left_player))


def get_faces(image_gray, face_cascade):
    ''' Detect faces in the photo '''
    faces = face_cascade.detectMultiScale(image_gray,
                                          scaleFactor=1.1,
                                          minNeighbors=8,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    return faces


def group_faces(faces, img_shape):
    ''' Splits face frames into left and right subsets
    Requires faces returned by face cascade akgirithm from OpenCV '''
    left, right = [], []

    for face in faces:
        if (face[0]+face[2]//2) < img_shape[0]//2:
            left.append(face)
        else:
            right.append(face)

    return left, right


class SmilesState:
    def __init__(self, faces, faces_left, faces_right, faces_both, y_hat=None):
        if y_hat is None:
            y_hat = []
        self.faces = faces
        self.faces_left = faces_left
        self.faces_right = faces_right
        self.faces_both = faces_both
        self.y_hat = y_hat
