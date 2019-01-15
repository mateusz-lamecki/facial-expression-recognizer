from timeit import default_timer as timer
import random
import sys

from PIL import Image, ImageDraw
import numpy as np
import cv2

from common import IM_SIZE, LABELS
import utils

PLAYER1_LED = 13
PLAYER2_LED = 19


class Game:
    FRAMES_COLOR = (255, 255, 255)
    SECS_PER_EXPRESSION = 12
    CERTAINTY_THRES = .3
    RESOLUTION = (480, 320)
    FRAMES_PER_SMILE_EVAL = 5
    GAME_TITLE = 'Nieszczere emocje - gra'

    def __init__(self, model, face_cascade, rpi_only):
        self.model = model
        self.face_cascade = face_cascade
        self.points = [0, 0]
        self.rpi_only = rpi_only

        cv2.namedWindow(self.GAME_TITLE, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.GAME_TITLE, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    def run(self):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.RESOLUTION[0]//2)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.RESOLUTION[1]//2)

        last_time = timer()
        choices = sorted(LABELS, key=lambda x: random.random())
        n_label = 0
        i = 0

        while n_label < len(choices):
            ret_val, img_raw = cam.read()
            img_raw = cv2.resize(img_raw, dsize=self.RESOLUTION,
                                 interpolation=cv2.INTER_CUBIC)


            smile_eval = (i%self.FRAMES_PER_SMILE_EVAL == 0)
            img_with_labels = self.__game_frame(img_raw, choices[n_label],
                                                smile_eval)

            cv2.imshow(self.GAME_TITLE, img_with_labels)

            if cv2.waitKey(1) == 27:
                sys.exit(0)

            if timer()-last_time > self.SECS_PER_EXPRESSION:
                last_time = timer()
                n_label += 1

            i += 1

        cv2.destroyAllWindows()

        winner = self.points[0] < self.points[1]
        self.__show_winner(winner)

        cv2.destroyAllWindows()



    def __game_frame(self, img_raw, expression, smile_eval):
        img_raw = np.flip(img_raw, axis=1)
        img_raw_bw = np.dot(img_raw[..., :3],
                            [0.299, 0.587, 0.114]).astype(np.uint8)

        img_colored = Image.fromarray(img_raw)

        if smile_eval:
            faces = utils.get_faces(img_raw_bw, self.face_cascade)
            faces_left, faces_right = utils.group_faces(faces, img_colored.size)
            faces_both = [(x, 0) for x in faces_left] + [(x, 1) for x in
                                                         faces_right]

            if self.rpi_only:
                ''' Control LEDs '''
                import RPi.GPIO as GPIO

                nonempty = lambda x: len(x) > 0
                if(nonempty(faces_left) != nonempty(self.prev_state.faces_left)):
                    GPIO.output(PLAYER1_LED, GPIO.HIGH if nonempty(faces_left)
                                else GPIO.LOW)
                if(nonempty(faces_right) != nonempty(self.prev_state.faces_right)):
                    GPIO.output(PLAYER1_LED, GPIO.HIGH if nonempty(faces_right)
                                else GPIO.LOW)


            self.prev_state = utils.SmilesState(faces, faces_left,
                                                faces_right, faces_both)
        else:
            prev_state = self.prev_state

        expression_i = LABELS.index(expression)

        points_new = [0, 0]

        for i, ((x, y, w, h), player) in enumerate(self.prev_state.faces_both):
            ImageDraw.Draw(img_colored).rectangle([(x, y), (x + w, y + h)],
                                                  fill=None,
                                                  outline=self.FRAMES_COLOR)
            if smile_eval:
                face_arr = img_raw_bw[y:y+h, x:x+w]
                face_arr = Image.fromarray(face_arr, 'L')
                img_resized = face_arr.resize((48, 48), Image.ANTIALIAS)
                X = (np.array(img_resized) / 255.).reshape(1, *IM_SIZE, 1)

                y_hat = self.model.predict(X).ravel()
                self.prev_state.y_hat.append(y_hat)
            
            certainty = int(np.round(self.prev_state.y_hat[i][expression_i] * 100, -1))
            certainty_str = str(certainty) + '%'
            img_colored = utils.draw_text(img_colored, certainty_str, (x+w, y),
                                          self.FRAMES_COLOR, right_side=False,
                                          text_size=w//20)

            if certainty >= self.CERTAINTY_THRES:
                points_new[player] += certainty

        for i, x in enumerate([self.prev_state.faces_left, self.prev_state.faces_right]):
            if len(x) > 0:
                points_new[i] /= len(x)
            self.points[i] += points_new[i] / 10

        img_colored = self.__draw_game_shapes(img_colored, self.prev_state.faces, expression)

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

    def __show_winner(self, winner):
        winner_window_title = 'And the winner is...'

        while True:
            key = cv2.waitKey(1)
            if key == 27:
                sys.exit(0)
            elif key == 32: 
                break

            img = Image.new('RGB', self.RESOLUTION)
            label = 'Wygrał gracz ' + ('lewy' if winner == 0 else 'prawy')
            img = utils.draw_text(img, label, (self.RESOLUTION[0]//2,
                                               self.RESOLUTION[1]//2),
                                  (255, 255, 255), center=True, text_size=25)

            img = np.array(img).astype(np.uint8)

            cv2.namedWindow(winner_window_title, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(winner_window_title, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
            cv2.imshow(winner_window_title, img)
