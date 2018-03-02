import spaceman
import os.path
import cv2

def main():
    c = spaceman.Face('test/cheinrich.jpg')
    c.detect_face()
    rotated_face = c.get_face(-30)
    cv2.imshow('face', rotated_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
