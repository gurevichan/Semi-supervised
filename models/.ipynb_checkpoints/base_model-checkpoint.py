import os
from abc import ABC
from contextlib import redirect_stdout

import tensorflow as tf


# TODO: create baseclassification model class
# from augmentations.random_padding import ZeroPaddingAugmentation
import numpy as np

class ZeroPaddingAugmentation(tf.keras.layers.Layer):
    def __init__(self, height, width, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width

    def call(self, inputs, training=None):
        if not training:
            return inputs

        inputs = tf.keras.layers.ZeroPadding2D(padding=(self.height, self.width))(inputs)

        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "height": self.height,
            "width": self.width,
        })
        return config

class BaseModel(ABC):
    def __init__(self, config, output_dir, input_shape, n_classes, optimizer, weight_decay=0):
        self.config = config
        self.output_dir = output_dir
        self.checkpoints_dir = os.path.join(self.output_dir, 'checkpoints')
        self.keras_json_file = os.path.join(self.checkpoints_dir, 'keras.json')

        self._set_loss_fn()
        self._set_augmentations(input_shape)
        self._set_model(input_shape, n_classes, optimizer)
        self.set_weight_decay(weight_decay)
        self.save_model_json()
        self._save_model_summary()

    def save_model_json(self):
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        model_json = self.model.to_json()
        with open(self.keras_json_file, 'w') as json_file:
            json_file.write(model_json)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def _set_loss_fn(self):
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def _set_model(self, input_shape, n_classes, optimizer, *args, **kwargs) -> None:
        # self.model = tf.keras.Sequential([
        #     self.data_augmentation,
        #     self.model_class(input_shape=input_shape, weights=None, pooling=None, classes=n_classes)
        # ])
        # self.model.compile(optimizer=optimizer, loss=self.loss_fn, metrics=['accuracy'])
        self.model = tf.keras.Sequential([])
        if self.data_augmentation:
            self.model.add(self.data_augmentation)
        self.model.add(self.model_class(input_shape=input_shape, weights=None, pooling=None, classes=n_classes))
        self.model.compile(optimizer=optimizer, loss=self.loss_fn, metrics=['accuracy'])

    def _save_model_summary(self):
        with open(os.path.join(self.output_dir, 'model_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                for layer in self.model.layers:
                    layer.summary()

    def _set_augmentations(self, input_shape):
        valid_augmentations = {
            "random_flip": (tf.keras.layers.experimental.preprocessing.RandomFlip, (["horizontal_and_vertical"])),
            "random_crop": (tf.keras.layers.experimental.preprocessing.RandomCrop, (32, 32)),
            "resize": (tf.keras.layers.experimental.preprocessing.Resizing, (32, 32)),
            "padding": (ZeroPaddingAugmentation, (4, 4)),
            "normalization": (tf.keras.layers.experimental.preprocessing.Normalization, (-1, (0.4914, 0.4822, 0.4465), (np.square(0.247), np.square(0.243), np.square(0.261))))
            # (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        }

        self.data_augmentation = tf.keras.Sequential([])
        if self.config["data_augmentation"]:
            for augmentation in self.config["data_augmentation"]:
                aug_class, aug_args = valid_augmentations[augmentation]
                self.data_augmentation.add(aug_class(*aug_args, input_shape=input_shape))
        else:
            self.data_augmentation = None

    def set_weight_decay(self, alpha=0):

        if alpha > 0:
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                    layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(alpha)(layer.kernel))
                if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                    layer.add_loss(lambda layer=layer: tf.keras.regularizers.l2(alpha)(layer.bias))
