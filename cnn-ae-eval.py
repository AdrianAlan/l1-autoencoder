import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import setGPU
import tensorflow as tf

from sklearn.metrics import roc_curve, auc


def score(truth, predictions):
    truth = truth.reshape(predictions.shape)
    score = np.sum(np.sum(np.sum((truth - predictions), 3), 2), 1)
    return score


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Train the L1 Tower CNN AE")
    parser.add_argument('dataset_test_inlier',
                        type=str,
                        help='Inlier dataset')
    parser.add_argument('dataset_test_outlier',
                        type=str,
                        help='Outlier dataset')
    parser.add_argument('path', type=str, help='Path to trained model')
    parser.add_argument('-b', '--batch-size', type=int, default=100,
                        help='Number of training samples in a batch',
                        dest='batch_size')
    args = parser.parse_args()

    # Prepare the dataset
    X_inlier = np.load(args.dataset_test_inlier)
    X_outlier = np.load(args.dataset_test_outlier)

    inlier_dataset = tf.data.Dataset.from_tensor_slices(X_inlier) \
        .batch(args.batch_size)
    outlier_dataset = tf.data.Dataset.from_tensor_slices(X_outlier) \
        .batch(args.batch_size)

    # Load the model
    model = tf.keras.models.load_model(args.path)

    # Model summary
    model.summary()

    # Feed forward pass for test samples
    inlier_predictions = model.predict(inlier_dataset)
    outlier_predictions = model.predict(outlier_dataset)

    plt.figure()
    plt.subplot(211)
    plt.imshow(inlier_predictions[0].reshape(56, 72), cmap=plt.cm.BuPu_r)
    plt.title("Truth")
    plt.subplot(212)
    plt.title("Reconstruction")
    plt.imshow(X_inlier[0].reshape(56, 72), cmap=plt.cm.BuPu_r)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.savefig("example.png")
    plt.close()

    inlier_score = score(X_inlier, inlier_predictions)
    outlier_score = score(X_outlier, outlier_predictions)
    scores = np.append(inlier_score, outlier_score)
    truths = np.append(np.zeros(len(inlier_score)),
                       np.ones(len(outlier_score)))

    fpr, tpr, _ = roc_curve(truths, scores)
    auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr,
             tpr,
             color='darkorange',
             lw=2,
             label='ROC curve (area = %0.2f)' % auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig("roc.png")
    plt.close()
