import tensorflow as tf
import pickle

# Creating PreProcess class
class PreProcess:
    def __init__(self):
        self.files = None
        self.images = None
    
    def get_path(self, file_path):
        self.files = tf.constant(file_path)

    def standardize(self):
        image_string = tf.read_file(self.files)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [64, 64])
        self.images = tf.image.per_image_standardization(image_resized)
        return self.images
  
    def transform(self):
        transformed_data = tf.data.Dataset.from_tensor_slices(self.files)
        transformed_data = transformed_data.map(self.standardize())

        return transformed_data

image_transformer = PreProcess()

with open('transformer', 'wb') as file:
    pickle.dump(image_transformer , file)


# prep = PreProcess(file_path)
# prep = prep.standardize()
# data = prep.transform()