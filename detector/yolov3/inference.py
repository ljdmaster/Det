from yolo3.model import Yolo, YoloTiny
from yolo3.inference import Infer
from PIL import Image
import os
model_dir = os.path.abspath("./yolo3/model/trained/yolov3.h5")
anchors_path = "./yolo3/model/yolo_anchors.txt"
classes_path = './yolo3/model/coco_classes.txt'



yolo = Yolo()
infer = Infer(yolo, model_dir, anchors_path, classes_path)
'''
img = './VOC2012/JPEGImages/2011_000054.jpg'
img = Image.open(img)
r_img = infer.detect_image(img)
r_img.show()
'''
infer.detect_video(0)
