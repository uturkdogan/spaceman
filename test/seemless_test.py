import spaceman
import os.path
import cv2

def main():
    bg = cv2.imread('test/spaceman.jpg')
    c = spaceman.Face('test/cheinrich.jpg')
    c.detect_face()
    clone = c.seemless_copy_face(bg, -30)
    cv2.imshow('face', clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
