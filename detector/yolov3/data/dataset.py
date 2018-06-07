import xml.etree.ElementTree as ET
from os import getcwd
import os
import shutil


def read_classes(classes_file):
    classes = []
    with open(classes_file) as f:
        line = f.readline()
        while line:
            line = line.strip()
            classes.append(line)
            line = f.readline()
    return classes



def convert_annotation(in_file, classes, list_file):
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def convert_all_annotation(data_dir, ratio):
    data_dir = os.path.abspath(data_dir)
    anno_dir = os.path.join(data_dir, "Annotations")
    image_dir = os.path.join(data_dir, "JPEGImages")
    classes_file = os.path.join(data_dir,"classes.txt")

    image_ids = [name.split('.')[0] for name in os.listdir(anno_dir)]
    image_num = len(image_ids)

    classes = read_classes(classes_file)
    
    with open("annotations.txt",'w') as list_file:
        for image_id in image_ids[:int(image_num*ratio)]:
            list_file.write(image_dir+"/"+image_id+".jpg")
            convert_annotation(anno_dir+'/'+image_id+".xml", classes, list_file)
            list_file.write('\n')
    list_file.close()
    
    with open("test_image.txt", 'w') as test_image:
        for image_id in image_ids[int(image_num*ratio):]:
            test_image.write(image_dir+"/"+image_id+".jpg")
            test_image.write('\n')
    test_image.close()

    shutil.copyfile(classes_file, "classes.txt")
    
    

if __name__=="__main__":
    #data_dir = "../VOC2012"
    data_dir = "../FaceinCar2"
    convert_all_annotation(data_dir, 0.997)

