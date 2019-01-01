from timeit import default_timer as timer
import random

from PIL import Image, ImageDraw
import numpy as np
import cv2

from common import IM_SIZE, LABELS
import utils


class Game:
    FRAMES_COLOR = (255, 255, 255)
    SECS_PER_EXPRESSION = 10
    CERTAINTY_THRES = .3

    def __init__(self, model, face_cascade):
        self.model = model
        self.face_cascade = face_cascade
        self.points = [0, 0]

    def run(self):
        cam = cv2.VideoCapture(0)

        last_time = timer()
        choices = sorted(LABELS, key=lambda x: random.random())
        n_label = 0

        while n_label < len(choices):
            ret_val, img_raw = cam.read()

            img_with_labels = self.__game_frame(img_raw, choices[n_label])
            cv2.imshow('Face expression', img_with_labels)
            if cv2.waitKey(1) == 27: 
                break

            if timer()-last_time > self.SECS_PER_EXPRESSION:
                last_time = timer()
                n_label += 1


        cv2.destroyAllWindows()


    def __game_frame(self, img_raw, expression):
        img_raw = np.flip(img_raw, axis=1)
        img_raw_bw = np.dot(img_raw[..., :3],
                            [0.299, 0.587, 0.114]).astype(np.uint8)

        img_colored = Image.fromarray(img_raw)
        faces = utils.get_faces(img_raw_bw, self.face_cascade)
        faces_left, faces_right = utils.group_faces(faces, img_colored.size)
        faces_both = [(x, 0) for x in faces_left] + [(x, 1) for x in
                                                     faces_right]

        expression_i = LABELS.index(expression)

        points_new = [0, 0]

        for (x, y, w, h), player in faces_both:
            ImageDraw.Draw(img_colored).rectangle([(x, y), (x + w, y + h)],
                                                  fill=None,
                                                  outline=self.FRAMES_COLOR)
            face_arr = img_raw_bw[y:y+h, x:x+w]
            face_arr = Image.fromarray(face_arr, 'L')
            img_resized = face_arr.resize((48, 48), Image.ANTIALIAS)
            X = (np.array(img_resized) / 255.).reshape(1, *IM_SIZE, 1)

            y_hat = self.model.predict(X).ravel()
            certainty = int(np.round(y_hat[expression_i] * 100, -1))
            certainty_str = str(certainty) + '%'
            img_colored = utils.draw_text(img_colored, certainty_str, (x+w, y),
                                          self.FRAMES_COLOR, right_side=False,
                                          text_size=w//20)

            if certainty >= self.CERTAINTY_THRES:
                points_new[player] += certainty

        for i, x in enumerate([faces_left, faces_right]):
            if len(x) > 0:
                points_new[i] /= len(x)
            self.points[i] += points_new[i] / 10

        img_colored = self.__draw_game_shapes(img_colored, faces, expression)
        img_colored = cv2.resize(img_colored, (1280, 960), cv2.INTER_CUBIC)

        return img_colored


    def __draw_game_shapes(self, img, faces, expression):
        ''' Draws game shapes
        Requires PIL.Image on input
        Returns image as np.array '''

        HEADER_HEIGHT = 30

        left, right = utils.group_faces(faces, img.size)

        img = np.array(img).astype(np.uint8)

        img_with_box = img.copy()
        cv2.rectangle(img_with_box,
                      (0, 0),
                      (img.shape[1], HEADER_HEIGHT),
                      (0, 0, 0), -1)
        cv2.addWeighted(img_with_box, .5, img, .5, 0, img)

        img = Image.fromarray(img)

        for i, x in enumerate([left, right]):
            img = utils.print_player_status(img, len(x), self.points[i],
                                            self.FRAMES_COLOR, (i == 0))

        img = utils.draw_text(img, expression, (img.size[0]//2, 0),
                              self.FRAMES_COLOR, center=True, text_size=20)
        img = np.array(img).astype(np.uint8)

        cv2.line(img,
                 (0, HEADER_HEIGHT),
                 (img.shape[1], HEADER_HEIGHT),
                 self.FRAMES_COLOR, 3)

        cv2.line(img,
                 (img.shape[1]//2, HEADER_HEIGHT),
                 (img.shape[1]//2, img.shape[1]),
                 self.FRAMES_COLOR, 3)

        return img

