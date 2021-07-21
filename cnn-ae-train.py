import argparse
import numpy as np
import os
import setGPU
import tensorflow as tf


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Train the L1 Tower CNN AE")
    parser.add_argument('dataset_train', type=str, help='Path to dataset')
    parser.add_argument('dataset_validation', type=str, help='Path to dataset')
    parser.add_argument('save_path', type=str, help='Path to trained model')
    parser.add_argument('-b', '--batch-size', type=int, default=100,
                        help='Number of training samples in a batch',
                        dest='batch_size')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='Number of training epochs', dest='epochs')
    parser.add_argument('-s', '--steps', type=int, default=1000,
                        help='Number of steps per epoch', dest='steps')
    parser.add_argument('-p', '--patience', type=int, default=2,
                        help='LR reduction callback patience',
                        dest='patience')
    parser.add_argument('-w', '--workers', type=int, default='10',
                        help='Number of workers', dest='workers')
    args = parser.parse_args()

    # Prepare the dataset
    X_train = np.load(args.dataset_train)
    X_validation = np.load(args.dataset_validation)

    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    X_validation = tf.data.Dataset.from_tensor_slices(X_validation)

    train_dataset = tf.data.Dataset \
        .zip((X_train, X_train)) \
        .shuffle(60000).batch(args.batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    validation_dataset = tf.data.Dataset \
        .zip((X_validation, X_validation)) \
        .shuffle(60000).batch(args.batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
