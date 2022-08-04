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
import wandb

import models
import utils

mirrored_strategy = tf.distribute.MirroredStrategy() # TODO: finish this


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


class SupervisedTrainer():

    def __init__(self, model_type, decay_steps, lr, num_classes=10, train_data_fraction=1.0, resume=False, wandb=None, **kwargs):
        with mirrored_strategy.scope():
            self.model = utils.create_model(model_type, num_classes)
        self.categorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy()
        learning_rate_fn = tf.keras.experimental.CosineDecay(lr, decay_steps=decay_steps)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
        self.weight_decay = 5e-4
        self.wandb = wandb 

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        self.ckpt_path = f'./checkpoints/{model_type}/train_frac_{train_data_fraction}'
        self.resume = resume

    @tf.function  #(jit_compile=True)
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            # Cross-entropy loss
            ce_loss = self.categorical_cross_entropy(labels, predictions)
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

    def train(self, train_ds, test_ds, epoch, **kwargs):
        best_acc, curr_epoch, manager = self._init_train_step()

        for e in tqdm(range(int(curr_epoch), epoch)):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for images, labels in train_ds:
                self.train_step(images, labels)

            for images, labels in test_ds:
                self.test_step(images, labels)
            
            self._log(e)
            self.save_checkpoint(best_acc, curr_epoch, manager, epoch=e)
    
    def reset_states(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

    def _init_train_step(self):
        best_acc = tf.Variable(0.0)
        curr_epoch = tf.Variable(0)  # start from epoch 0 or last checkpoint epoch
        checkpoint = tf.train.Checkpoint(curr_epoch=curr_epoch, best_acc=best_acc, optimizer=self.optimizer, model=self.model)
        manager = tf.train.CheckpointManager(checkpoint, self.checkpoint_path, max_to_keep=1)

        if self.resume:
            print('==> Resuming from checkpoint...')
            assert os.path.isdir(self.checkpoint_path), 'Error: no checkpoint directory found!'
            checkpoint.restore(manager.latest_checkpoint)
        return best_acc,curr_epoch,manager
    
    def _log(self, e):
        template = f"Epoch {e+1}, Loss: {self.train_loss.result():.4f}, Accuracy: {self.train_accuracy.result()*100:.2f}%, " + \
                   f"Test Loss: {self.test_loss.result():.4f}, Test Accuracy: {self.test_accuracy.result()*100:.2f}%"
        print(template)
        if self.wandb is not None:
            self.wandb.log({'train_loss': self.train_loss.result(), 'train_accuracy': self.train_accuracy.result()*100,
                            'test_loss': self.test_loss.result(), 'test_accuracy': self.test_accuracy.result()*100})
            

    def save_checkpoint(self, best_acc, curr_epoch, manager, epoch):
        # Save checkpoint
        if self.test_accuracy.result() > best_acc:
            print('Saving...')
            if not os.path.isdir(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            best_acc.assign(self.test_accuracy.result())
            curr_epoch.assign(epoch + 1)
            manager.save()

    def predict(self, pred_ds, best, model=None):
        model = self.model if model is None else model
        if best:
            # TODO: this loads the best model INPLACE!!!!
            utils.load_weights_to_model(model, self.ckpt_path)
        utils.evaluate_model(self.model, pred_ds)


def main(args):
    train_ds, test_ds, decay_steps = utils.prepare_data(args.train_data_fraction, args.batch_size * mirrored_strategy.num_replicas_in_sync, args.epoch)
    # Train

    print('==> Building model...')
    wandb.init(project="my-test-project")
    wandb.config.update(args)
    wandb.config.update({"tain_class": "SupervisedTrainer"})
    
    trainer = SupervisedTrainer(args.model, decay_steps, lr=args.lr, num_classes=10, 
                                train_data_fraction=args.train_data_fraction, resume=args.resume, wandb=wandb)


    trainer.train(train_ds, test_ds, args.epoch)

    # Evaluate
    trainer.predict(test_ds, best=True)
    # TODO: create script evaluate checkpoint
    # TODO: evaluate teacher
    # TODO: save teacher origin to checkpoint dir
    # TODO: create save checkpoint function that saves also the current accuracy and epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TensorFlow2.0 CIFAR-10 Training')
    parser.add_argument('--model', default='resnet18', type=str, help='model type')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int, help='number of training epoch')
    parser.add_argument('--train_data_fraction', default=1.0, type=float, help='fraction of data to use in train')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--gpu', default=0, type=int, help='specify which gpu to be used')
    args = parser.parse_args()
    args.model = args.model.lower()

    main(args)
