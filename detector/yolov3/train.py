from yolo3.model import Yolo
from yolo3.train import Train


annotation_path = "data/image_label.txt"
log_dir = "yolo3/logs/000"
classes_path = "data/classes.txt"
anchors_path =  "yolo3/model/yolo_anchors.txt"
weights_path = "yolo3/model/yolov3.h5"

yolo = Yolo()
tr = Train(yolo, annotation_path,log_dir,classes_path,anchors_path, weights_path)
tr.train()
