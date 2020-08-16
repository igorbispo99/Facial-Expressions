import numpy as np
import os
import shutil
import re
import cv2 as cv

from cv2 import CascadeClassifier, imread, imwrite
from glob import glob

faces_hash = {'0':"neutral", '1':"joy", '2':"sadness",\
    '3':"surprise", '4':"angry", '5':"disgust", '6':"fear"}

dataset_folder = './facesdb/'
ext = 'bmp'
cl = CascadeClassifier("haarcascade_frontalface_default.xml")

def gen_croped_img(img_path, out_path):
    img = imread(img_path)

    faces = detect_faces_cascade(img)
    croped = crop_face(img, faces)

    imwrite(out_path, croped)

def detect_faces_cascade(img, color = (0, 0, 255)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = cl.detectMultiScale(gray, 1.3, 5)

    return faces

def crop_face(img, faces):
    # Assuming that exists only one face in the image
    (x, y, w, h) = faces[0]

    croped_img = img[y:y+w, x:x+h]

    return croped_img

def gen_data():
    try:
        os.mkdir("./data")
    except FileExistsError:
        pass

    folder_size_dict = {}

    for key, val in faces_hash.items():
        folder_size_dict[val] = 0
        try:
            os.mkdir("./data/{}".format(val))
        except FileExistsError:
            pass


    for folder in os.listdir(dataset_folder):
        cur_folder = dataset_folder + folder + '/' + ext 
        for img in glob(cur_folder + "/*." + ext):
            exp_code = re.search("-[0-9][0-9]_", img).group()[2]

            folder_name = faces_hash[exp_code]
            cur_name = folder_size_dict[folder_name]
            gen_croped_img(img, './data/' + folder_name + '/' + str(cur_name) + "." + ext)

            folder_size_dict[folder_name] += 1


gen_data()