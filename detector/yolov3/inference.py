from yolo3.model import Yolo, YoloTiny
from yolo3.inference import Infer
from PIL import Image
import os
import time
import numpy as np
import cv2

model_dir = os.path.abspath("./yolo3/model/trained/yolov3.h5")
anchors_path = "./yolo3/model/yolo_anchors.txt"
classes_path = './data/mcoco/classes.txt'




yolo = Yolo()
infer = Infer(yolo, model_dir, anchors_path, classes_path)

image_dir = './FaceinCar2/JPEGImages'
image_list = os.listdir(image_dir)

for image_name in image_list[:2]:
    image_file = os.path.join(image_dir, image_name)
    img = Image.open(image_file)

    r_img = infer.detect_image(img)
    r_img = np.asarray(r_img)
    cv2.imshow("image", r_img)
    cv2.waitKey(2000)

#infer.detect_video(0)
