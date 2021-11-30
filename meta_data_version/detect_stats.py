import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pytesseract
import re
import sys

path = '../Pawpularity-Contest/data/train/0a05c55ca864b667d31c80ce2c68d6b3.jpg'

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
    face_cascade = cv2.CascadeClassifier('data/haarcasdade_cat.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return 0
    else:
        return 1

def detect_dogs(path_image):
    img = cv2.imread(path_image)
    face_cascade = cv2.CascadeClassifier('data/haarcasdade_dog.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return 0
    else:
        return 1

def detect_cats_groups(path_image):
    img = cv2.imread(path_image)
    face_cascade = cv2.CascadeClassifier('data/haarcasdade_cat.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) < 1:
        return 0
    else:
        return 1

def detect_dogs_groups(path_image):
    img = cv2.imread(path_image)
    face_cascade = cv2.CascadeClassifier('data/haarcasdade_dog.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) < 1:
        return 0
    else:
        return 1

def detect_collage(path_image):
    gray = cv2.imread(path_image)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return 0
    else:
        return 1

def focal_lng(m_distantce,real_width,width_in_rf_image):
    return (width_in_rf_image*m_distantce) / real_width

def distance(focal_lng,rf_width,object_width):
    return (rf_width*focal_lng)/object_width

def pet_data(image,isDog):
    pet_width = 0
    if isDog:
        pet_detector= cv2.CascadeClassifier('data/haarcasdade_dog.xml')
    else:
        pet_detector = cv2.CascadeClassifier('data/haarcasdade_cat.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pets = pet_detector.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, h, w) in pets:
        cv2.rectangle(image, (x, y), (x+w, y+h), WHITE, 1)
        pet_width = w

    return pet_width

def detect_focus(path_image):
    img = cv2.imread(path_image)
    eyes_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye = eyes_cascade.detectMultiScale(gray, 1.3, 5)
    if len(eye) == 0:
        return 0
    else:
        return 1

def detect_text(path_image):
    img = cv2.imread(path_image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    invert = 255 - opening 
    text=pytesseract.image_to_string(invert, lang='eng')
    if len(text) == 0:
        return 0
    else:
        return 1

def detect_occluded(path_image):
    image= cv2.imread(path_image)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[np.where((image <= [15,15,15]).all(axis=2))] = [255,255,255]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    percentage = (cv2.countNonZero(mask)/ (w * h)) * 100
    if percentage < 2:
        return 0
    else:
        return 1


def detect_stats(path_image):
    model_cnn = tf.keras.models.load_model('/data/photo_version/model.h5')
    subject_focus = detect_focus(path_image)
    blur_img = detect_blur(path_image)
    collage_image = detect_collage(path_image)
    is_humans = detect_humans(path_image)
    oclussion = detect_occluded(path_image)

    image = cv2.imread(path_image)
    image = cv2.resize(image, (150, 150))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(1,150, 150, 1)
    predict = model_cnn.predict([image/255.0])
    if predict[0][0] > 0.5:
        isDog=True
    else:
        isDog=False
    
    if isDog:
        face,eyes = detect_dogs(path_image),detect_focus(path_image)
        group = detect_dogs_groups(path_image)
    else:
        face,eyes = detect_cats(path_image),detect_focus(path_image)
        group = detect_cats_groups(path_image)
    
    accesories = 0
    info = detect_text(path_image)
    actions = 0
    near = 0
    model = tf.keras.models.load_model('data/model.h5')
    predict = model.predict([[subject_focus,eyes,face,near,actions,accesories,group,collage_image,is_humans,oclussion,info,blur_img]])
    return predict

sys.modules[__name__] = detect_stats