'''
    This script is designed to adequate the dataset to the tensorflow
    input format so that will be possible to build a classification
    model with tensorflow.

    After adequation, a model is builted using the keras API.
'''

import tensorflow as tf
from tensorflow.keras import layers, Model, metrics
# from keras.models import Sequential
# from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
# from keras.optimizers import SGD, Adam
# from keras import regularizers
import glob

DATASET_SIZE = 1203
BATCH_SIZE = 8
TRAIN_SIZE = 960
NUM_CLASSES = 2

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
labels = tf.one_hot(tf.cast(labels, tf.int32), NUM_CLASSES)

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
    test_data = test_data.batch(batch_size=BATCH_SIZE)
    test_data = test_data.repeat()

    return train_data, test_data

dataset = create_and_shuffle_dataset(filenames,labels)
train_data, test_data = preparing_for_training(dataset)

# --------- MODEL -------------

# Inputs
inputs = tf.keras.Input(shape=(64,64,3))

# Convolutional Layer 1
conv1 = layers.Conv2D(filters=32, kernel_size=[5,5], padding='same', activation='relu')(inputs)

# Pooling Layer 1
pool1 = layers.MaxPool2D(pool_size=(2,2), strides=2)(conv1)

# Convolutional Layer 2
conv2 = layers.Conv2D(filters=64, kernel_size=[3,3], activation=tf.nn.relu)(pool1)

# Pooling Layer 2
pool2 = layers.MaxPool2D(pool_size=(2,2), strides=2)(conv2)

# Flattening
pool2_flat = layers.Flatten()(pool2)

# Dense Layer 1
dense1 = layers.Dense(512, activation=tf.nn.relu)(pool2_flat)

# Dropping out with a probability of 'rate'
dropped = layers.Dropout(rate=0.4)(dense1)

# output Layer
predictions = layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)(dropped)

model = Model(inputs=inputs, outputs=predictions)

# -------- MODEL PARAMETERS ---------

# Instantiating an ADAM Optimizer
# sgd = tf.train.GradientDescentOptimizer(learning_rate=0.1)
adam = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999)

model.compile(loss=metrics.binary_crossentropy, optimizer=adam, metrics=[metrics.categorical_accuracy])

H = model.fit(
    dataset, 
    epochs=8,
    steps_per_epoch=120,
    validation_data=test_data,
    validation_steps = 40
)

# print(H.history)