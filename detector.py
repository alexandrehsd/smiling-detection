'''
    This script is designed to adequate the dataset to the tensorflow
    input format so that will be possible to build a classification
    model with tensorflow.

    After adequation, a model is built using the keras API.
'''

import tensorflow as tf
from tensorflow.keras import layers, Model, metrics
import glob
from random import shuffle

DATASET_SIZE = 1203
BATCH_SIZE = 16
TRAIN_SIZE = 960
NUM_EPOCHS = 16
NUM_CLASSES = 2

# --------------------------------------------------------------------
#           ADEQUATING THE DATASET TO TENSORFLOW OBJECTS
# --------------------------------------------------------------------

filenames_list = []
labels_list = []

filenames_list = glob.glob("./lfwcrop_color/labeled_faces/*.jpg")
shuffle(filenames_list)

# last image of the non-smile list
divisor = './lfwcrop_color/labeled_faces/Jacques_Chirac_0001.jpg'

#  Mounting the filenames and the labels list
for i in range(len(filenames_list)):
    if filenames_list[i] > divisor:
        labels_list.append(1)
    else:
        labels_list.append(0)

# A vector of filenames.
filenames = tf.constant(filenames_list)

# 'labels[i]' is the label for the image in 'filenames[i]'.
labels = tf.constant(labels_list)
labels = tf.one_hot(tf.cast(labels, tf.int32), NUM_CLASSES)

# --------------------------------------------------------------------
#           PREPROCESSING THE DATA AND CREATING A DATASET
# --------------------------------------------------------------------

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image_resized = tf.image.resize_images(image_decoded, [64, 64])
  std_image = tf.image.per_image_standardization(image_resized)
  return std_image, label

def create_dataset(filenames, labels):
    # Generating tf.data.Dataset object and shuffling it
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)

    return dataset

def train_test_split(dataset):
    # defining batch size and 'count' number of epochs
    # taking ~ 80% for training and ~ 20% for testing
    train_data = dataset.take(TRAIN_SIZE) 
    train_data = train_data.batch(batch_size=BATCH_SIZE)
    train_data = train_data.repeat()

    test_data = dataset.skip(TRAIN_SIZE)
    test_data = test_data.batch(batch_size=BATCH_SIZE)
    test_data = test_data.repeat()

    return train_data, test_data

dataset = create_dataset(filenames,labels)
train_data, test_data = train_test_split(dataset)

# --------------------------------------------------------------------
#                       BUILDING THE MODEL
# --------------------------------------------------------------------

# Inputs
inputs = tf.keras.Input(shape=(64,64,3))

# Convolutional Layer 1
conv1 = layers.Conv2D(filters=32, kernel_size=[3,3], padding='same', activation=tf.nn.relu)(inputs)

# Pooling Layer 1
pool1 = layers.MaxPool2D(pool_size=(2,2), strides=2)(conv1)

# Convolutional Layer 2
conv2 = layers.Conv2D(filters=64, kernel_size=[3,3], activation=tf.nn.relu)(pool1)

# Pooling Layer 2
pool2 = layers.MaxPool2D(pool_size=(2,2), strides=2)(conv2)

# Convolutional Layer 2
conv3 = layers.Conv2D(filters=128, kernel_size=[3,3], activation=tf.nn.relu)(pool2)

# Pooling Layer 2
pool3 = layers.MaxPool2D(pool_size=(2,2), strides=2)(conv3)

# Flattening
pool3_flat = layers.Flatten()(pool3)

# Dense Layer 1
dense1 = layers.Dense(512, activation=tf.nn.relu)(pool3_flat)

# Dropping out with a probability of 'rate'
dropped = layers.Dropout(rate=0.5)(dense1)

# output Layer
predictions = layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)(dropped)

# Instantiating a tensorflow Model object
model = Model(inputs=inputs, outputs=predictions)

# ADAM Optimizer
adam = tf.train.AdamOptimizer(learning_rate=0.001)

model.compile(loss=metrics.binary_crossentropy, optimizer=adam, metrics=[metrics.categorical_accuracy])
model.summary()

# --------------------------------------------------------------------
#                       FITTING THE MODEL
# --------------------------------------------------------------------

H = model.fit(
    train_data, 
    epochs=NUM_EPOCHS,
    steps_per_epoch=60,
    validation_data=test_data,
    validation_steps=15
)

# print(H.history)

# --------------------------------------------------------------------
#                       EVALUATING THE MODEL
# --------------------------------------------------------------------

files_ls = []
files_ls = glob.glob("./lfwcrop_color/evaluation_faces/not_smiling/*.jpg")

label_ls = [0,0,0,0,0,0,0,0,0]

# A vector of filenames.
samples = tf.constant(files_ls)

# 'labels[i]' is the label for the image in 'filenames[i]'.
label = tf.constant(label_ls)
label = tf.one_hot(tf.cast(label, tf.int32), NUM_CLASSES)

eval_data = create_dataset(samples, label)
eval_data = eval_data.batch(batch_size=BATCH_SIZE)

result = model.predict(eval_data, steps=1)
print(result)