from __future__ import print_function

import tensorflow as tf
import os

base_path = 'C:\\Users\\Maith\\Documents\\Datasets\\Dataset_food-101\\images3'

def read_data():
    image_list = []
    label_list = []
    label_map_dict = {}
    count_label = 0

    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        label_map_dict[class_name] = count_label

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            label_list.append(count_label)
            image_list.append(image_path)

        count_label += 1
    return image_list, label_list, label_map_dict




def _parse_function(filename, label):

    image_string = tf.read_file(filename, "file_reader")
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)

    return image, label

image_list, label_list, label_map_dict = read_data()

dataset = tf.data.Dataset.from_tensor_slices((tf.constant(image_list), tf.constant(label_list)))
dataset = dataset.shuffle(len(image_list))
dataset = dataset.repeat(2)
dataset = dataset.map(_parse_function).batch(32)

dataset = dataset.map(_parse_function)

iterator = dataset.make_one_shot_iterator()


print(dataset)
print(len(dataset))
