import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
path = 'data/train/00b151a572c9aabedf8cfce0fa18be25.jpg'

def detect_blur(path_image):
    img = cv2.imread(path_image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_blur_diff = cv2.absdiff(img_gray, img_blur)
    _, img_thresh = cv2.threshold(img_blur_diff, 20, 255, cv2.THRESH_BINARY)
    img_thresh_dilate = cv2.dilate(img_thresh, None, iterations=3)
    contours, _ = cv2.findContours(img_thresh_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    else:
        return 1

def detect_humans(path_image):
    img = cv2.imread(path_image)
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return 0
    else:
        return 1
    

def detect_cats(path_image):
    img = cv2.imread(path_image)
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return 0
    else:
        return 1

def detect_dogs(path_image):
    img = cv2.imread(path_image)
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalcatface.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return 0
    else:
        return 1

def detect_collage(path_image):
    gray = cv2.imread(path_image)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    plt.imshow(edges)
    plt.show()
    if lines is None:
        return 0
    else:
        return 1

