## clone https://github.com/0kt0pus/ml_utils.git and install using setup.py
## The dataset can be downloaded from http://host.robots.ox.ac.uk/pascal/VOC/
from ml_utils.utils import function_utils as fu
import xml.etree.ElementTree as ET
import os
import random

import tensorflow as tf
from PIL import Image
import numpy as np

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    #if isinstance(value, type(tf.constant(0))):
    #    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


## make the feature dictionary
def image_example(image_string, label, image_shape):
    #image_shape = tf.image.decode_jpeg(image_string).shape
    #image_shape = tf.shape(image_string)
    #print(image_string)
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }
    #print(feature)
    return tf.train.Example(features=tf.train.Features(feature=feature))

def get_single_object_image(annotation_dir, image_dir):
    
    xml_paths = fu.get_xml_paths(annotation_dir)
    ## open the annotation files and count the number of objects
    ## list to hold the number of objects in an image
    num_obj_list = list()
    obj_list = list()
    for xfile in xml_paths:
        xml_keys = ET.parse(xfile)
        root = xml_keys.getroot()
        ## get the objects
        class_list = list()
        for obj in root.findall('object'):
            name = obj.find('name')
            class_list.append(name.text)
        ## store the number of objects in a single image
        num_obj_list.append(len(class_list))
        obj_list.append(class_list)

    '''
    ## filter any file that has more than 1 object
    for i, elem in enumerate(num_obj_list):
        if elem == 1:
            single_obj_list.append(xml_paths[i])
    '''
    single_obj_xml_list = [xml_paths[i] for i, elem in enumerate(num_obj_list) if elem == 1]
    single_obj_img_path_list = [os.path.join(image_dir, '.'.join([xml_path.split('/')[-1].split('.')[0], 'jpg'])) for xml_path in single_obj_xml_list]
    single_obj_list = [obj_list[i] for i, elem in enumerate(num_obj_list) if elem == 1]
    
    ## make a dict of object file list
    label_image_dict = dict()
    for obj, img in zip(single_obj_list, single_obj_img_path_list):
        if obj[0] in label_image_dict:
            label_image_dict[obj[0]].append(img)
        else:
            label_image_dict[obj[0]] = [img]
    #print(single_obj_img_path_list)
    '''
    for k, v in label_image_dict.items():
        print(k, v)
    '''
    return single_obj_img_path_list, single_obj_list, label_image_dict

def open_label_file(label_file_path):
    label_name_list = list()
    with open(label_file_path, 'r') as f:
        for line in f:
            label_name_list.append(line.split("\n")[0])
    return label_name_list

def write_tfrecord_classification(record_path, image_path_list, class_list, label_file_path, lb, ub, label_image_dict):
    ## open the label file and make an array of labels
    ## list to hold the label names
    label_name_list = open_label_file(label_file_path)
    num_labels = len(label_name_list)
    ## list that holds the number of images to take for each class
    num_sampels_per_label = [np.random.randint(lb, ub, 1) for _ in range(num_labels)]
    #print(num_sampels_per_label)
    ## make a new dict that updates the number of sampels per class
    new_label_image_dict = {k: v[:num_sampels_per_label[i][0]] for i, (k, v) in enumerate(label_image_dict.items())}
    #print(new_label_image_dict)
    ## Now extract the dict to a tuple with (class, path) pair
    label_image_tuple_list = list()
    for k, v in new_label_image_dict.items():
        for item in v:
            label_image_tuple_list.append((k, item))
    ## shuffle the tuple list
    random.shuffle(label_image_tuple_list)
    #print(label_image_tuple_list)
    ## generate a set of random numbers between lb and ub
    #print(label_name_list)
    
    with tf.io.TFRecordWriter(record_path) as writer:
        ## take a image and corresponding class and add to the tfrecord file
        ## Iterate over the image paths and open them
        for lbl_name, img_path in label_image_tuple_list:
            #print(lbl_name)
            ## open the image
            image = np.array(Image.open(img_path))
            #print(image.shape)
            image_shape = image.shape
            ## get the label index
            label_idx = encode_label(label_name_list, lbl_name)
            bin_image = image.tobytes('C')
            #print(bin_image)
            tf_example = image_example(bin_image, label_idx, image_shape)
            writer.write(tf_example.SerializeToString())
            #print(label_idx)
    

def encode_label(label_name_list, label_name):
    #print(label_name_list)
    #print(label_name)
    ## get the index of the label from the label name list
    return label_name_list.index(label_name)
    

anno_dir = 'VOCdevkit/VOC2012/Annotations'
img_dir = 'VOCdevkit/VOC2012/JPEGImages'
label_file = './labels.txt'
record_path = './pascalvoc_max_32_set.tfrecords'

img_paths, lbls, img_lbl_dict = get_single_object_image(anno_dir, img_dir)
write_tfrecord_classification(record_path, img_paths, lbls, label_file, 2, 32, img_lbl_dict)

        


