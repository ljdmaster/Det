from yolo3.model import Yolo, YoloTiny
from yolo3.infer import Infer
from common import detect
from config import test_cfgs

from PIL import Image
import os
import time
import numpy as np
import cv2
from datetime import datetime

from common import statistics, detect




yolo = Yolo()
infer = Infer(yolo, **test_cfgs)


def detect_test(image_dir):
    image_list = os.listdir(image_dir)

    for image_name in image_list:
        image_file = os.path.join(image_dir, image_name)

        start = datetime.now()
        img = Image.open(image_file)   
        r_img = detect.detect_image(infer, img)  # detect face in car
        end = datetime.now()
        print("Image {}, time diff: {}".format(image_name, (end-start).total_seconds()))
    
        #show
        r_img = np.asarray(r_img)
        if r_img.ndim == 3:
            r_img = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", r_img) 
        cv2.waitKey(2000)




def eval_detector(annotation_path):
    with open(annotation_path) as f:
        lines = f.readlines()

    pred_count = 0
    real_count = 0
    right_count = 0

    for line in lines:
        line = line.strip('\n').split(' ')
        image_file = line[0]

        box = []
        temp = 0
        for b in line[1:]:        
            bx_ = b.split(',')[:4]
            bx = []
            bx.append(float(bx_[1]))
            bx.append(float(bx_[0]))
            bx.append(float(bx_[3]))
            bx.append(float(bx_[2]))
            box.append(bx)
            temp+=1
        print("real_count: ", temp)
        real_count += temp

        box= np.array(box)

        # show label 
        label_img = Image.open(image_file)
        colors = detect.get_colors(1)
        label = 'face'
        for bx in box:
            draw = detect.draw_label(label_img, label, bx, colors)
            del draw
        cv2.imshow("label", np.array(label_img))
    

        # count
        pred_img = Image.open(image_file)
        out_boxes, _, _ = infer.detect_image(pred_img)
        num_boxes = len(out_boxes)
        pred_count += num_boxes
        if num_boxes != 0:
            count = statistics.count(out_boxes, box)
            right_count += count        
            print("right count: ", count)

    
        # show predict
        r_img = detect.detect_image(infer, pred_img)
        r_img = np.asarray(r_img)
        if r_img.ndim == 3:
            r_img = cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("pred", r_img)
    
        cv2.waitKey(2000)
        print("\n")



    print("pred_count: ",pred_count)
    print("real_count: ", real_count)
    print("right_count: ", right_count)

    print(1.0*right_count/real_count)


if __name__=='__main__':
    #image_dir = './FaceinCar/JPEGImages'
    #detect(image_dir)

    annotation_path = './data/faceincar/vals.txt'
    eval_detector(annotation_path)
