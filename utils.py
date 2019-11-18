import numpy as np
import torch
import helper_func

def pairwise_distance(X,Y):
    N, d_y = X.shape
    M, d_x = Y.shape
    assert d_y == d_x

    D = np.zeros((N,M))

    for i in range(N):
        D[i] = np.linalg.norm(X[i]-Y[:], axis=1)
    return np.array(D)

def nearest_neighbor_classify(train_image_feat, train_labels, test_image_feats, k=3):
    test_labels = []

    labels = list(set(train_labels))

    label_idx = np.arange(len(labels))

    m_dict = {}
    idx_training_labels = []

    for key, value in zip(labels, label_idx):
        m_dict[key] = value
    for i in range(len(train_labels)):
        idx_training_labels.append(m_dict[train_labels[i]])
    

    D = pairwise_distance(test_image_feats, train_image_feat)
    np_training_labels = np.array(idx_training_labels, dtype=int)
    nearest_idx = np.argsort(D, axis=1)[:, :k]

    for i in range(test_image_feats.shape[0]):
        nearest_labels = np_training_labels[nearest_idx[i]]
        test_labels.append(labels[np.argmax(np.bincount(nearest_labels))])

    return test_labels