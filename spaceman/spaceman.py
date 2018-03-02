from .detect_face import Face
import os, sys
from cv2 import imwrite, imread

class Spaceman:
    def __init__(self):
        pass

    @staticmethod
    def generate_image(input, output):
        assert os.path.isfile(input) and not os.path.isfile(output), "Check if file exists"
        image = Face(input)
        image.detect_face()
        spaceman = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'spaceman.jpg')
        spaceman_img = imread(spaceman)
        face = image.seemless_copy_face(spaceman_img, (722, 406), (120, 120), -25)
        imwrite(output, face)

def main():
    assert len(sys.argv) > 1, "No arguements"
    input = ''
    output = ''
    if len(sys.argv) > 1:
        input = sys.argv[1]
    if len(sys.argv) > 2:
        output = sys.argv[1]
    else:
        output = '.'.join(input.split('.')[:-1]) + '_output.' + (input.split('.')[-1])
    if not os.path.isabs(input):
        input = os.path.join(os.getcwd(), input)
        output = os.path.join(os.getcwd(), output)
    Spaceman.generate_image(input, output)

if __name__ == '__main__':
    main()
