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
from train import SupervisedTrainer, _init_args
import utils

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

class MeanTeacher(SupervisedTrainer):

    def __init__(self, model_type, decay_steps, lr, num_classes=10, train_data_fraction=1.0, 
                 resume=False, wandb=None, consistency_weight=1., ema_start=0.99):
        super(MeanTeacher, self).__init__(model_type, decay_steps, lr, num_classes=num_classes, 
                                          train_data_fraction=train_data_fraction, resume=resume, wandb=wandb)
        self.checkpoint_path = f'./checkpoints/{model_type}/MeanTeacher/train_frac_{train_data_fraction}'
        self.teacher = copy.deepcopy(self.model)
        self.ema_start = ema_start
        self.l2_loss = tf.keras.losses.MeanSquaredError()
        self.consistency_loss_metric = tf.keras.metrics.Mean(name='consistency_loss_metric')
        self.test_accuracy_teacher = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_teacher')
        self.consistency_weight = consistency_weight

    def ema_teacher_weights(self, ema_alpha):
        for teacher_weight, model_weight in zip(self.teacher.trainable_variables, self.model.trainable_variables):
            teacher_weight.assign(ema_alpha * teacher_weight + (1 - ema_alpha) * model_weight)

    def train(self, train_ds, test_ds, epoch, unlabeled_ds, **kwargs):
        best_acc, curr_epoch, manager = self._init_train_step()

        for e in tqdm(range(int(curr_epoch), epoch)):
            training_progress = e/float(epoch)
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.consistency_loss_metric.reset_states()

            for (images, labels), unlabeled_imgs in zip(train_ds, unlabeled_ds):
                self.train_step(images, labels, unlabeled_imgs, training_progress=training_progress)

            for images, labels in test_ds:
                self.test_step(images, labels)
            
            self._log(e)
            # self.save_checkpoint(best_acc, curr_epoch, manager, epoch=e)
    
    @tf.function
    def train_step(self, images, labels, unlabeled_imgs, training_progress):
        u1 = unlabeled_imgs
        u2, _ = utils._augment_fn(unlabeled_imgs, {}, adjust_colors=True)
        u1_pred = self.teacher(u1, training=True)
        consistency_weight, ema_alpha = self._get_weight_decay_and_ema_alpha(training_progress)
        with tf.GradientTape() as tape:
            labeled_preds = self.model(images, training=True)
            # u1_pred = self.teacher(u1, training=True)
            u2_preds = self.model(u2, training=True)

            supervised_loss = self.categorical_cross_entropy(labels, labeled_preds)
            consistency_loss = self.l2_loss(u1_pred, u2_preds, sample_weight=1)
            weight_decay_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])

            loss = supervised_loss + weight_decay_loss * self.weight_decay + consistency_loss * consistency_weight
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, labeled_preds)
        self.ema_teacher_weights(ema_alpha)
        self.consistency_loss_metric(consistency_loss)

    def _get_weight_decay_and_ema_alpha(self, training_progress, verbose=False):
        """
        training_progress is between (0,1)
        """
        start_consistency_weight = 0.0
        end_consistency_weight = self.consistency_weight
        end_ema_alpha = 0.995
        consistency_weight = (1 - training_progress) * start_consistency_weight + training_progress * end_consistency_weight
        ema_alpha = (1 - training_progress) * self.ema_start + training_progress * end_ema_alpha
        if verbose:
            print(f'{training_progress=:.2f}, {consistency_weight=:.2f}, {ema_alpha=:.4f}')
        return consistency_weight, ema_alpha

    def _log(self, e):
        super()._log(e)
        print(f'Consistency loss: {self.consistency_loss_metric.result():.4f}, ' \
              f'Test Accuracy Teacer: {self.test_accuracy_teacher.result()*100:.2f}%')
        if self.wandb:
            self.wandb.log({'consistency_loss': self.consistency_loss_metric.result(),
                            'test_accuracy_teacher': self.test_accuracy_teacher.result()*100})


    @tf.function
    def test_step(self, images, labels):
        super().test_step(images, labels)
        predictions = self.teacher(images, training=False)
        self.test_accuracy_teacher(labels, predictions)

def main():
    parser, wandb = _init_args()
    parser.add_argument('--consistency_weight', '-cw', default=1, type=float, help='specify consistency weight')
    parser.add_argument('--unlabled_bs_multiplier', '-bsm', default=2, type=int, help='specify batch size multiplier for unlabled data, will increase unlabled data fraction')
    parser.add_argument('--ema', default=0.99, type=float, help='ema weight to start with')
    args = parser.parse_args()
    args.model = args.model.lower()
    wandb.config.update(args)
    wandb.config.update({"tain_class": "MeanTeacher"})
    if args.additional_wandb_args is not None:
        wandb.config.update({f"additional arg {i}": arg for i, arg in enumerate(args.additional_wandb_args)})

    train_ds, test_ds, unlabeled_ds, decay_steps = utils.prepare_data(args.train_data_fraction, args.batch_size, 
                                                                      args.epoch, get_unlabeled=True, unlabled_bs_multiplier=args.unlabled_bs_multiplier)

    print('==> Building model...')
    trainer = MeanTeacher(args.model, decay_steps, lr=args.lr, num_classes=10, 
                          train_data_fraction=args.train_data_fraction, resume=args.resume, 
                          wandb=wandb, consistency_weight=args.consistency_weight, ema_start=args.ema)
    trainer.train(train_ds, test_ds, args.epoch, unlabeled_ds=unlabeled_ds)
    # Evaluate
    utils.evaluate_model(trainer.model, test_ds)


if __name__ == "__main__":
    main()
