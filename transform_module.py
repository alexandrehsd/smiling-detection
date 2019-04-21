import tensorflow as tf
import glob

# Creating PreProcess class
class preProcess:
    def __init__(self):
        self.files = None
        self.images = None
        self.num_files = 0
    
    def get_images(self, file_path):
        self.files = sorted(glob.glob(file_path))
        self.num_files = len(self.files)
        self.files = tf.constant(self.files)

    def standardize(self, files):
        image_string = tf.read_file(files)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [64, 64])
        self.images = tf.image.per_image_standardization(image_resized)

        return self.images
  
    def transform(self):
        transformed_data = tf.data.Dataset.from_tensor_slices(self.files)
        transformed_data = transformed_data.map(self.standardize)
        transformed_data = transformed_data.batch(batch_size=self.num_files)

        return transformed_data