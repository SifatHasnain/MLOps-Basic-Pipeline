import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import yaml
import sys
import logging
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy 
from tqdm import tqdm

class Val:
    def __init__(self, testloader, inception=False):
        self.testloader = testloader
        self.inception = inception

    def loss(self, model, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        if self.inception:
            y_, _, _ = model(x, training=training)
        else:
            y_ = model(x, training=training)
            
        loss_object = SparseCategoricalCrossentropy(from_logits=False)

        return loss_object(y_true=y, y_pred=y_)

    
    def run(self, model, callbacks):
        # with torch.no_grad():
        #     model.eval()
            # epoch_loss = 0.0
            logs = {}
            epoch_loss_avg = tf.keras.metrics.Mean()
            # correct = 0
            # total = 0
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            bar = tqdm(enumerate(self.testloader), total=len(self.testloader), desc="Val: ")
            for batch_id, data in bar:
                callbacks.on_batch_begin(batch_id, logs=logs)
                callbacks.on_test_batch_begin(batch_id,logs=logs)
                inputs, labels = data[0], data[1]
                # outputs = model(inputs)
                loss = self.loss(model, inputs, labels, training=False)
                epoch_loss_avg.update_state(loss) 
                # epoch_loss += float(loss.item())
                # _, predicted = torch.max(outputs.data, 1)
                # prediction = tf.math.argmax(outputs, axis=1, output_type=tf.int64)
                test_accuracy.update_state(labels, model(inputs, training=False))

                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                bar.set_postfix_str('Loss='+str(tf.keras.backend.get_value(epoch_loss_avg.result()))+', Accuracy:'+str(tf.keras.backend.get_value(test_accuracy.result())))
                logs['val_loss'] = float(test_accuracy.result())
                callbacks.on_test_batch_end(batch_id)
                callbacks.on_batch_end(batch_id)
            return tf.keras.backend.get_value(epoch_loss_avg.result()), float(test_accuracy.result()) #/len(bar), round(100*correct/total, 4)
    
    