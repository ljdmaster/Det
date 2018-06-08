import os

from yolo3.model import Yolo
from yolo3.train import Train

data_dir = "./data/faceincar2"



annotation_path = os.path.join(data_dir, "annotations.txt")
classes_path = os.path.join(data_dir, "classes.txt")
anchors_path =  "./yolo3/model/yolo_anchors.txt"
weights_path = "./yolo3/model/trained/yolov3.h5"
log_dir = "./yolo3/logs"

yolo = Yolo()
tr = Train(yolo, annotation_path, classes_path, anchors_path, weights_path, log_dir)
tr.train()
