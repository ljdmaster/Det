#/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import random
from timeit import default_timer as timer
from PIL import ImageFont, ImageDraw
import numpy as np


def get_colors(num_classes):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x/num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def convert_box(box, image_size):
    top, left, bottom, right = box[0], box[1], box[2], box[3]
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image_size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image_size[0], np.floor(right + 0.5).astype('int32'))
    return top, left, bottom, right



def draw_label(image, label, box, colors):
    thickness = (image.size[0] + image.size[1]) // 400
    font = ImageFont.truetype(font='./common/font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    draw = ImageDraw.Draw(image)
    
    label_size = draw.textsize(label, font)
    
    top, left, bottom, right = convert_box(box, image.size)
    print(label, (left, top), (right, bottom))

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    # My kingdom for a good redistributable image drawing library.
    if np.asarray(image).ndim==3:
        outline = colors[0]
        fill = (0,0,0)
    elif np.asarray(image).ndim==2:
        outline = 128
        fill = 0
    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i],outline=outline)
    draw.rectangle([tuple(text_origin), tuple(text_origin +label_size)],fill=outline)
    draw.text(text_origin, label, fill=fill, font=font)
    
    return draw


def draw_image(infer, image, out_boxes, out_scores, out_classes, colors):
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = infer.class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = draw_label(image, label, box, colors)  
        del draw
    return image



def detect_image(infer, image):
    out_boxes, out_scores, out_classes = infer.detect_image(image)
    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    colors = get_colors(infer.num_classes)
    image = draw_image(infer, image, out_boxes, out_scores, out_classes, colors)
    return image


def detect_video(infer, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    infer.close_session()



