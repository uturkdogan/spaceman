import spaceman
import os.path

def main():
    c = spaceman.Face('test/cheinrich.jpg')
    c.detect_face()
    c.show_debug_image()

main()
