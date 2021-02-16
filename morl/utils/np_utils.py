
import numpy as np

__all__ = ['np_softmax', 'np_cross_entropy']


def np_softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def np_cross_entropy(probs, labels):
    if labels.shape[-1] == 1:
        # sparse label
        n_classes = probs.shape[-1]
        result_shape = list(labels.shape[:-1]) + [n_classes]
        labels = np.eye(n_classes)[labels.reshape(-1)]
        labels = labels.reshape(result_shape)

    return -np.sum(labels * np.log(probs), axis=-1, keepdims=True)
