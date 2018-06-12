import sys
sys.path.append("..")

# test imdb

# test pascal_voc
from pascal_voc import pascal_voc

imdb = pascal_voc("train", "2007")
print(imdb._year)
print(imdb._image_set)
print(imdb._devkit_path)
print(imdb._data_path)
print(imdb._classes)
print(imdb._class_to_ind)
#print(imdb._image_ext)
#print(imdb._image_index)
print(imdb.cache_path)
gt_roidb = imdb.gt_roidb()
imdb.rpn_roidb()
