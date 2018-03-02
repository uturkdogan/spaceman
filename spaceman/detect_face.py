import numpy as np
import cv2, math, imutils
import os.path

class Face:
    '''
    Uses and image and finds face and crops it from background
    '''
    FACE_CASCADE = cv2.CascadeClassifier('spaceman/haarcascade_frontalface_alt.xml')
    EYE_CASCADE = cv2.CascadeClassifier('spaceman/haarcascade_eye.xml')

    def __init__(self, image):
        self.angle = 0
        self.face = []
        self.eyes = []
        self.marked_image = None
        self.image = cv2.imread(image)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def get_angle(self, roi, eyes):
        angle = 0
        (x1,y1,w1,h1) = eyes[0]
        (x2,y2,w2,h2) = eyes[1]
        (x_mid1, y_mid1) = (x1 + w1/2, y1 + h1/2)
        (x_mid2, y_mid2) = (x2 + w2/2, y2 + h2/2)
        angle = math.atan2((y_mid2 - y_mid1), (x_mid2 - x_mid1))
        return angle

    def detect_face(self):
        '''
        Gets the first face from image
        returns bool according to success
        '''
        faces = self.FACE_CASCADE.detectMultiScale(self.gray_image, 1.1, 5)
        success = False
        if len(faces) > 0:
            self.face = faces[0]
            (x,y,w,h) = self.face
            face_roi = self.gray_image[y:y+h, x:x+w]
            eyes = self.EYE_CASCADE.detectMultiScale(face_roi)
            if len(eyes) == 2:
                self.eyes = eyes
                self.angle = self.get_angle(face_roi, eyes)
                success = True
        return success

    def get_face(self, rotate_to=0, size=None):
        '''
        get face cutout and rotate it to given Angle
        size [width, height]
        '''
        assert type(rotate_to) is int, 'Angle is not integer: {0}'.format(rotate_to)
        assert (size == None
                or (len(size) == 2
                    and type(size[0]) is int
                    and type(size[1]) is int)), "Size is incorrect"
        if self.face != [] and len(self.eyes) == 2:
            rotation = rotate_to - self.angle
            (x,y,w,h) = self.face
            face = self.image.copy()[y:y+h, x:x+w]
            face = imutils.rotate(face, rotation)
            print(size)
            if size != None:
                face = imutils.resize(face, width=size[0], height=size[1])
            return face
        else:
            raise Exception("No face data is available, Run detect_face first.")

    def show_debug_image(self):
        if self.face != [] and len(self.eyes) == 2:
            (x,y,w,h) = self.face
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            eye_mid_points = []
            for (ex,ey,ew,eh) in self.eyes:
                cv2.rectangle(img[y:y+h, x:x+w], (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                eye_mid_points.append((int(ex + ew/2), int(ey + eh/2)))
            cv2.line(img[y:y+h, x:x+w], eye_mid_points[0], eye_mid_points[1], (0,0,255), 1)
            print('Eye Angle:', self.angle)
            cv2.imshow('face', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            raise Exception("No face data is available, Run detect_face first.")

    def seemless_copy_face(self, background, center=None, size=None, rotate_to=0):
        '''
        put face on background image
        '''
        face = self.get_face(rotate_to, size=size)
        width, height, channels = background.shape
        if center == None:
            center = (int(height/2), int(width/2))
        mask = 255 * np.ones(face.shape, face.dtype)  # All black mask
        mixed_clone = cv2.seamlessClone(face, background, mask, center, cv2.NORMAL_CLONE)
        return mixed_clone
