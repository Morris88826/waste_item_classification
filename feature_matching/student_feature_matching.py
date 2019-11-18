import numpy as np


def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # raise NotImplementedError('`match_features` function in ' +
    #     '`student_feature_matching.py` needs to be implemented')
    dists = np.zeros((features1.shape[0], features2.shape[0]))
    for i,feat in enumerate(features1):
        dists[i,:] = np.linalg.norm(features2 - feat[np.newaxis, :], axis=1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # raise NotImplementedError('`match_features` function in ' +
    #     '`student_feature_matching.py` needs to be implemented')

    matches = []
    confidences = []
    distances = compute_feature_distances(features1, features2)
    for i, f1 in enumerate(features1):
        dists = distances[i,:]
        order = np.argsort(dists)
        if dists[order[0]] / dists[order[1]] < 0.75:
            matches.append([i, order[0]])
            confidences.append(dists[order[0]])

    print('Found matches: ', len(matches))
    matches = np.asarray(matches, dtype=np.int)
    confidences = np.asarray(confidences)
    confidences /= np.max(confidences)

    # sort matches and confidences by decreasing confidence (doesn't affect
    # result)
    order = np.argsort(-confidences)
    matches = matches[order]
    confidences = confidences[order]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
