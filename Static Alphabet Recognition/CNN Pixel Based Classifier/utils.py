import tensorflow as tf
import os

IMG_SIZE = 64  # you can increase this if you want

def preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0  # normalize to [0,1]
    return img

def load_dataset(data_dir):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        label_mode='int',
        shuffle=True,
        seed=123
    )
    return dataset