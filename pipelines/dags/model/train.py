import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import os
import yaml
import sys
import logging
from tensorflow.keras.utils import Progbar
from tensorflow.keras.losses import SparseCategoricalCrossentropy 
# print(tf.executing_eagerly())
class Trainer:
    def __init__(self, train_loader, inception=False):
        # self.device = device
        self.train_loader = train_loader
        self.inception = inception
        # l = loss(model, features, labels, training=False)
        # print("Loss test: {}".format(l))
    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def loss(self, model, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        loss_object = SparseCategoricalCrossentropy(from_logits=False)
        
        if self.inception:
            y_, aux_1, aux_2 = model(x, training=training)
            loss1 = loss_object(y_true=y, y_pred=y_)
            loss2 = loss_object(y_true=y, y_pred=aux_1)
            loss3 = loss_object(y_true=y, y_pred=aux_2)

            return loss1 + 0.4*loss2 + 0.3*loss3
        else:
            y_ = model(x, training=training)
            return loss_object(y_true=y, y_pred=y_)

    def run(self, model, optimizer, callbacks):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        metrics = {'acc': 0.0, 'loss': 0.0, 'val_acc': 0.0, 'val_loss': 0.0}
        # bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Train: ")
        progress_bar = Progbar(len(self.train_loader), stateful_metrics=list(metrics.keys()))
        
        # tf.executing_eagerly()
        print(tf.executing_eagerly())
        for batch_id, data in enumerate(self.train_loader):
            callbacks.on_batch_begin(batch_id)
            callbacks.on_train_batch_begin(batch_id)
    
            inputs, labels = data[0], data[1]
            
            # start training
            loss, grads = self.grad(model, inputs, labels)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # w = optimizer.get_weights()
            # Track progress
            epoch_loss_avg.update_state(loss)  
            # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(labels, model(inputs, training=True))
            # bar.set_postfix_str('Loss='+str(tf.keras.backend.get_value(epoch_loss_avg.result))) #/(batch_id+1)
            progress_bar.update(batch_id, values=[('acc',epoch_accuracy.result()),
                                       ('loss', loss)])
            # if batch_id==10:
            #     break
            callbacks.on_train_batch_end(batch_id)
            callbacks.on_batch_end(batch_id)
        return model, optimizer, epoch_loss_avg.result().numpy(), epoch_accuracy.result().numpy() #/len(bar)