# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" A simple MNIST classifier using tf.estimator.DNNClassifier """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


DEFAULT_FEATURE_NAME = "feature"
INPUT_SHAPE = [28, 28]
HIDDEN_UNITS = [256, 128]
MODEL_DIR = "output"
N_CLASSES = 10
DROPOUT = 0.5
LEARNING_RATE = 0.001
EPSILON = 1e-5
BATCH_SIZE = 100
MAX_STEPS = 300000
CHIEF = "chief"
TRAIN_EPOCH = 100
TEST_EPOCH = 1
EVAL_STEPS = 500
THROTTLE_SECS = 60
KEEP_CHECKPOINT_MAX = 3
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500
LOG_STEP_COUNT_STEPS = 500


def build_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.int)
    y_train = y_train.astype(np.int)
    x_test = x_test.astype(np.int)
    y_test = y_test.astype(np.int)
    return x_train, y_train, x_test, y_test


def build_config():
    return tf.estimator.RunConfig(
        keep_checkpoint_max=KEEP_CHECKPOINT_MAX,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        log_step_count_steps=LOG_STEP_COUNT_STEPS
    )


def build_feature_columns():
    return [tf.feature_column.numeric_column(DEFAULT_FEATURE_NAME, shape=INPUT_SHAPE)]


def build_classifier(feature_columns, run_config):
    return tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=HIDDEN_UNITS,
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=LEARNING_RATE, epsilon=EPSILON),
        n_classes=N_CLASSES,
        dropout=DROPOUT,
        model_dir=MODEL_DIR,
        config=run_config
    )


def build_input_fn():
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        {DEFAULT_FEATURE_NAME: x_train},
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True, num_epochs=TRAIN_EPOCH)

    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        {DEFAULT_FEATURE_NAME: x_test},
        y_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_epochs=TEST_EPOCH)

    return train_input_fn, eval_input_fn


def build_spec():
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=MAX_STEPS,
        hooks=[])

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=EVAL_STEPS,
        throttle_secs=THROTTLE_SECS)

    return train_spec, eval_spec


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity("INFO")

    run_config = build_config()
    x_train, y_train, x_test, y_test = build_mnist()
    feature_columns = build_feature_columns()
    classifier = build_classifier(feature_columns, run_config)
    train_input_fn, eval_input_fn = build_input_fn()
    train_spec, eval_spec = build_spec()

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    if run_config.task_type == CHIEF:
        classifier.evaluate(eval_input_fn)
