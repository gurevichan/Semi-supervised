"""
Some helper functions for TensorFlow2.0, including:
    - get_dataset(): download dataset from TensorFlow.
    - get_mean_and_std(): calculate the mean and std value of dataset.
    - normalize(): normalize dataset with the mean the std.
    - dataset_generator(): return `Dataset`.
    - progress_bar(): progress bar mimic xlua.progress.
"""
import tensorflow as tf
from tensorflow.keras import datasets

import numpy as np
import models
import sys 


padding = 4
image_size = 32
target_size = image_size + padding*2


def get_dataset(train_data_fraction=1.0):
    """Download, parse and process a dataset to unit scale and one-hot labels."""
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    assert train_data_fraction > 0.0 and train_data_fraction <= 1.0, "train_data_fraction must be in (0,1]"
    train_samples = int(len(train_labels) * train_data_fraction)
    train_images = train_images[:train_samples]
    train_labels = train_labels[:train_samples]
    print(f'Train labels distribution {np.unique(train_labels, return_counts=True)} for {train_samples} labeled images')
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images/255.0, test_images/255.0
    
    # One-hot labels
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)
    return train_images, train_labels, test_images, test_labels


def get_mean_and_std(images):
    """Compute the mean and std value of dataset."""
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    return mean, std


def normalize(images, mean, std):
    """Normalize data with mean and std."""
    return (images - mean) / std


def dataset_generator(images, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(len(images)).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def _one_hot(train_labels, num_classes, dtype=np.float32):
    """Create a one-hot encoding of labels of size num_classes."""
    return np.array(train_labels == np.arange(num_classes), dtype)


def _augment_fn(images, labels):
    images = tf.image.pad_to_bounding_box(images, padding, padding, target_size, target_size)
    images = tf.image.random_crop(images, (image_size, image_size, 3))
    images = tf.image.random_flip_left_right(images)
    return images, labels


def prepare_data(train_data_fraction, batch_size, epoch):
    # Data
    print('==> Preparing data...')
    train_images, train_labels, test_images, test_labels = get_dataset(train_data_fraction)
    mean, std = get_mean_and_std(train_images)
    train_images = normalize(train_images, mean, std)
    test_images = normalize(test_images, mean, std)
    
    train_ds = dataset_generator(train_images, train_labels, batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).\
            batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    decay_steps = int(epoch * len(train_images) / batch_size)
    return train_ds, test_ds, decay_steps


def evaluate_model(model, test_ds):
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    test_accuracy.reset_states()
    for images, labels in test_ds:
        test_step(model, test_accuracy, images, labels)
    print (f'Prediction Accuracy: {test_accuracy.result()*100:.2f}')


@tf.function
def test_step(model, test_accuracy, images, labels):
    predictions = model(images, training=False)
    test_accuracy(labels, predictions)


def create_model(model_type, num_classes):
    if 'lenet' in model_type:
        model = models.LeNet(num_classes)
    elif 'alexnet' in model_type:
        model = models.AlexNet(num_classes)
    elif 'vgg' in model_type:
        model = models.VGG(model_type, num_classes)
    elif 'resnet' in model_type:
        if 'se' in model_type:
            if 'preact' in model_type:
                model = models.SEPreActResNet(model_type, num_classes)
            else:
                model = models.SEResNet(model_type, num_classes)
        else:
            if 'preact' in model_type:
                model = models.PreActResNet(model_type, num_classes)
            else:
                model = models.ResNet(model_type, num_classes)
    elif 'densenet' in model_type:
        model = models.DenseNet(model_type, num_classes)
    elif 'mobilenet' in model_type:
        if 'v2' not in model_type:
            model = models.MobileNet(num_classes)
        else:
            model = models.MobileNetV2(num_classes)
    else:
        sys.exit(ValueError(f"{model_type} is currently not supported."))
    return model