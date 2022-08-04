"""Train CIFAR-10 with TensorFlow2.0."""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
import copy
from train import SupervisedTrainer, prepare_data
import utils


class DistillTrainer(SupervisedTrainer):
    def __init__(self, model_type, decay_steps, num_classes=10, teacher_ckpt=None):
        super(DistillTrainer, self).__init__(model_type, decay_steps, num_classes=10)
        self.ckpt_path = f'./checkpoints/{args.model}/distill/train_frac_{args.train_data_fraction}'
        self.teacher = copy.deepcopy(self.model)
        print(f'Teacher checkpoint {teacher_ckpt}')
        self.load_weights_to_model(self.teacher, teacher_ckpt)

    @tf.function #(jit_compile=True)
    def train_step(self, images, labels):
        teacher_pred = self.teacher(images)
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            # Cross-entropy loss
            ce_loss = self.categorical_cross_entropy(teacher_pred, predictions)
            # L2 loss(weight decay)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
            loss = ce_loss + l2_loss * self.weight_decay
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        t_loss = self.categorical_cross_entropy(labels, predictions)
        
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

        
def main(args):
    train_ds, test_ds, decay_steps = utils.prepare_data(args.train_data_fraction, args.batch_size, args.epoch)
    # Train
    print('==> Building model...')
    trainer = DistillTrainer(args.model, decay_steps, teacher_ckpt=args.teacher_ckpt)
    trainer.train(train_ds, test_ds, args.epoch)
    # Evaluate
    utils.evaluate_model(trainer.model, test_ds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TensorFlow2.0 CIFAR-10 Training')
    parser.add_argument('--model', required=True, type=str, help='model type')
    parser.add_argument('--teacher_ckpt', required=True, type=str, help='teacher checkpoint, assuming teacher is same model as student')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int, help='number of training epoch')
    parser.add_argument('--train_data_fraction', default=1.0, type=float, help='fraction of data to use in train')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--gpu', default=0, type=int, help='specify which gpu to be used')
    args = parser.parse_args()
    args.model = args.model.lower()

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    main(args)
