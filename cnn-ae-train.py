import argparse
import numpy as np
import os
import setGPU
import tensorflow as tf

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    # BatchNormalization,
    Conv2D,
    # Dropout,
    Dense,
    Flatten,
    Input,
    Reshape,
    UpSampling2D)


def CAE(input_size):
    inputs = Input(shape=input_size)
    x = Conv2D(16, kernel_size=(3, 3), use_bias=False, padding='same')(inputs)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(10, use_bias=False)(x)
    x = Dense(8064)(x)
    x = Activation('relu')(x)
    x = Reshape((14, 18, 32))(x)
    x = Conv2D(32, kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, kernel_size=(3, 3), use_bias=False, padding='same')(x)
    autoencoder = Model(inputs=inputs, outputs=x)
    return autoencoder


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
        .shuffle(60000) \
        .batch(args.batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    validation_dataset = tf.data.Dataset \
        .zip((X_validation, X_validation)) \
        .shuffle(60000) \
        .batch(args.batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    # Search for hyperparameters
    model = CAE((56, 72, 1))

    # Model summary
    model.summary()

    # Recompile the model
    loss = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss=loss, optimizer=optimizer)

    model.fit(train_dataset,
              epochs=args.epochs,
              validation_data=validation_dataset,
              callbacks=[EarlyStopping('val_loss',  patience=args.patience),
                         ModelCheckpoint(args.save_path)],
              use_multiprocessing=True,
              workers=args.workers)
