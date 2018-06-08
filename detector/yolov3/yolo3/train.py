"""
Retrain the YOLOv3 model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yolo3.utils import get_random_data



class Train(object):
    def __init__(self, yolo, 
                       annotation_path,
                       classes_path,
                       anchors_path,
                       weights_path,
                       log_dir='./yolo3/logs'):
        self.annotation_path = annotation_path
        self.classes_path = classes_path
        self.anchors_path = anchors_path
        self.weights_path = weights_path
        self.log_dir = log_dir

        self.class_names = self.get_classes()
        self.num_classes = len(self.class_names)
        self.anchors = self.get_anchors()
        self.annotations = self.get_annotations()
        
        self.input_shape = (416,416) # multiple of 32, hw
        self.batch_size = 16
        self.val_split = 0.1
        self.num_val = int(len(self.annotations)*self.val_split)
        self.num_train = len(self.annotations) - self.num_val
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(self.num_train, self.num_val, self.batch_size))

        self.model = yolo.create_model(self.input_shape,
                                        self.class_names,
                                        self.anchors,
                                        self.weights_path,
                                        load_pretrained=True,
                                        freeze_body=True)
   
    def get_classes(self):
        '''loads the classes'''
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        '''loads the anchors from a file'''
        with open(self.anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def get_annotations(self):
        with open(self.annotation_path) as f:
            lines = f.readlines()
        np.random.shuffle(lines)
        return lines

    def preprocess_true_boxes(self,true_boxes, input_shape, anchors, num_classes):
        '''Preprocess true boxes to training input format

        Parameters
        ----------
        true_boxes: array, shape=(m, T, 5)
            Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        input_shape: array-like, hw, multiples of 32
        anchors: array, shape=(N, 2), wh
        num_classes: integer

        Returns
        -------
        y_true: list of array, shape like yolo_outputs, xywh are reletive value

        '''
        num_layers = len(anchors)//3 # default setting
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

        m = true_boxes.shape[0]
        grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
            dtype='float32') for l in range(num_layers)]

        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes
        valid_mask = boxes_wh[..., 0]>0

        for b in range(m):
            # Discard zero rows.
            wh = boxes_wh[b, valid_mask[b]]
            # Expand dim to apply broadcasting.
            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = -box_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # Find best anchor for each true box
            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)
                        c = true_boxes[b,t, 4].astype('int32')
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5+c] = 1

        return y_true


    def data_generator(self, annotation_lines):
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        np.random.shuffle(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(self.batch_size):
                i %= n
                image, box = get_random_data(annotation_lines[i], self.input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i += 1
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
            yield [image_data, *y_true], np.zeros(self.batch_size)


    def data_generator_wrap(self, annotation_lines):
        n = len(annotation_lines)
        if n==0 or self.batch_size<=0: return None
        return self.data_generator(annotation_lines)


    
    def train(self):
        """retrain/fine-tune the model"""

        adam = Adam(lr=0.001)
        self.model.compile(optimizer=adam,loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        logging = TensorBoard(log_dir=self.log_dir)

        checkpoint = ModelCheckpoint(self.log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                     monitor='val_loss', 
                                     save_weights_only=True,
                                     save_best_only=True,
                                     period=500)

        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0, 
                                       patience=5, 
                                       verbose=1, 
                                       mode='auto')


        self.model.fit_generator(self.data_generator_wrap(self.annotations[:self.num_train]),
                                 steps_per_epoch=max(1, self.num_train//self.batch_size),
                                 validation_data=self.data_generator_wrap(self.annotations[self.num_train:]),
                                 validation_steps=max(1, self.num_val//self.batch_size),
                                 epochs=1000,
                                 initial_epoch=0,
                                 callbacks=[logging, checkpoint])

        self.model.save_weights(self.log_dir + 'yolo_weights.h5')



