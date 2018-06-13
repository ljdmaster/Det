import os


train_cfgs = {}
data_dir = './data/faceincar'
train_cfgs['classes_path'] = os.path.join(data_dir, "classes.txt")
train_cfgs['annotation_path'] = os.path.join(data_dir, "annotations.txt")
train_cfgs['anchors_path'] = './yolo3/model/yolo_anchors.txt'
train_cfgs['weights_path'] = './yolo3/model/trained/yolov3-faceincar-65000.h5'
train_cfgs['log_dir'] =  './yolo3/logs/'
train_cfgs['model_train_dir'] = './yolo/model/train'

train_cfgs['final_weights'] = 'yolov3_faceincar.h5'
train_cfgs['input_shape'] = (416,416)
train_cfgs['load_pretrained'] = True
train_cfgs['freeze_body'] = False
train_cfgs['epochs'] = 2500
train_cfgs['initial_epoch'] = 0
train_cfgs['batch_size'] = 8
train_cfgs['val_split'] = 0.1
train_cfgs['lr'] = 0.0001
train_cfgs['ckpt_period'] = 500




test_cfgs = {}
test_cfgs['anchors_path'] = "./yolo3/model/yolo_anchors.txt"
test_cfgs['model_path'] = os.path.abspath("./yolo3/model/trained/yolov3-faceincar-65000.h5")
test_cfgs['classes_path'] = './data/faceincar2/classes.txt'
test_cfgs['score_threshold'] = 0.6
test_cfgs['iou_threshold'] = 0.5
test_cfgs['cpu'] =  False
