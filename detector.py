'''
    This script is designed to adequate the dataset to the tensorflow
    input format so that will be possible to build a classification
    model with tensorflow.

    After adequation, a model is builted using the keras API.
'''

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
import glob

DATASET_SIZE = 1203
BATCH_SIZE = 8
TRAIN_SIZE = 960

filenames_list = []
labels_list = []

count = 1

# glob returns an unsorted list so that we need to sort list
sorted_names = sorted(glob.glob("./lfwcrop_color/labeled_faces/*.jpg"))

#  Mounting the filenames and the labels list
for file in sorted_names:
    filenames_list.append(file)
    if count <= 603: # the first 603 does not have a smile
        labels_list.append(0)
    else:
        labels_list.append(1)
    
    count += 1

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image_resized = tf.image.resize_images(image_decoded, [64, 64])
  return image_resized, label

# A vector of filenames.
filenames = tf.constant(filenames_list)

# 'labels[i]' is the label for the image in `filenames[i].
labels = tf.constant(labels_list)

def create_and_shuffle_dataset(filenames, labels):
    # Generating tf.data.Dataset object and shuffling it
    non_shuffled_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    non_shuffled_dataset = non_shuffled_dataset.map(_parse_function)
    
    dataset = non_shuffled_dataset.shuffle(buffer_size=DATASET_SIZE)

    return dataset

def preparing_for_training(dataset):
    # defining batch size and 'count' number of epochs
    # taking ~ 80% for training and ~ 20% for testing
    train_data = dataset.take(TRAIN_SIZE) 
    train_data = train_data.batch(batch_size=BATCH_SIZE)
    train_data = train_data.repeat()

    test_data = dataset.skip(TRAIN_SIZE)

    return train_data, test_data

dataset = create_and_shuffle_dataset(filenames,labels)
train_data, test_data = preparing_for_training(dataset)


