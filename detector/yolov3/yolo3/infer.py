#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import random
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from keras.backend import tensorflow_backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.utils import letterbox_image


class Infer(object):
    def __init__(self, yolo, **cfgs):
        self.yolo = yolo
        self.model_path = cfgs['model_path']
        self.anchors_path = cfgs['anchors_path']
        self.classes_path = cfgs['classes_path']
        
        self.score_threshold = cfgs['score_threshold']
        self.iou_threshold = cfgs['iou_threshold']
        
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.num_anchors = len(self.anchors)
        self.num_classes = len(self.class_names)
        #self.colors = self._colors()
        
        if cfgs['cpu']:
            config = tf.ConfigProto(device_count={"GPU":0})
            K.set_session(tf.Session(config=config))
        
        self.sess = K.get_session()

        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.input_image_shape = K.placeholder(shape=(2, ))
        
        self.yolo_model = self._load_weights()    
        self.boxes, self.scores, self.classes = self.yolo.yolo_eval(self.yolo_model.output,
                                                                    self.anchors,
                                                                    self.num_classes, 
                                                                    self.input_image_shape,
                                                                    score_threshold=self.score_threshold, 
                                                                    iou_threshold=self.iou_threshold)


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
   

    def _load_weights(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        try:
            yolo_model = load_model(model_path, compile=False)
        except:
            yolo_model = self.yolo.yolo_body(Input(shape=(None,None,3)),
                   self.num_anchors//3, self.num_classes) #
            yolo_model.load_weights(model_path) # make sure model, anchors and classes match
        else:
            assert yolo_model.layers[-1].output_shape[-1] == \
                   self.num_anchors/len(yolo_model.output) * (self.num_classes + 5), \
                   'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))
        return yolo_model
    
    ''''
    def _colors(self):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x/self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        return colors
    '''

    def output(self, image_data, input_image_shape):
        feed_dict={ self.yolo_model.input: image_data,
                    self.input_image_shape: input_image_shape,
                    K.learning_phase(): 0}
        out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes], feed_dict=feed_dict)
        return out_boxes, out_scores, out_classes


    def detect_image(self, image):
        start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        #print("image_data shape:", image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        #print("image_data:", image_data[0,100,100,0])
        out_boxes, out_scores, out_classes = self.output(image_data,[image.size[1], image.size[0]])
        
        #print("out_boxes:", out_boxes)
        return out_boxes, out_scores, out_classes


    def close_session(self):
        self.sess.close()


