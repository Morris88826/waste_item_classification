import numpy as np
import torch
from feature_matching.SIFTNet import get_siftnet_features
from utils import pairwise_distance


def kmeans(feature_vectors, k, max_iter = 100):
    # dummy centroids placeholder
    centroids = None

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    unique_center = np.unique(feature_vectors, axis=0)
#     print(feature_vectors)
#     print(feature_vectors[center_idx])
    centroids = unique_center[np.random.choice(unique_center.shape[0], size=k, replace=False)]

    for i in range(max_iter):

        cluster_idx = np.argmin(pairwise_distance(feature_vectors,centroids), axis=1)
        
        # print(pairwise_distances(feature_vectors,centroids))
        new_centers = np.array([[]])
        for c in range(centroids.shape[0]):
            mask = (cluster_idx == c)
            filtered = feature_vectors[mask]
            if len(filtered)==0:
                avg = np.expand_dims(feature_vectors[np.random.randint(feature_vectors.shape[0])], axis=0)
            else:
                avg = np.expand_dims(np.mean(filtered, axis=0), axis=0)
                

            if c==0:
                new_centers = avg
            else:
                new_centers = np.concatenate((new_centers, avg),axis=0)

        centroids = new_centers
        

#     raise NotImplementedError('kmeans function not implemented.')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return centroids


def build_vocabulary(image_arrays, vocab_size, stride = 20):
    
    dim = 128  # length of the SIFT descriptors that you are going to compute.
    vocab = None

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    features = []
    for image in image_arrays:
        width = image.shape[1]
        height = image.shape[0]
        interest_x = np.arange(10, width - 10, stride)
        interest_y = np.arange(10, height - 10, stride)
        interest = np.transpose([np.tile(interest_x, interest_y.shape[0]), np.repeat(interest_y, interest_x.shape[0])])
        x = interest[:,0]
        y = interest[:,1]

        image = torch.from_numpy(np.array(image, dtype=np.float32))
        img_tensor = image.reshape((1, 1, image.shape[0], image.shape[1]))
        sift_features = get_siftnet_features(img_tensor, x, y)
        features.extend(sift_features.flatten())


    features = np.array(features)

    features = features.reshape((features.shape[0]//dim, dim))

    vocab = kmeans(features, vocab_size)
#     raise NotImplementedError('build_vocabulary function not implemented.')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return vocab


def kmeans_quantize(raw_data_pts, centroids):

    indices = None

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    indices = np.argmin(pairwise_distance(raw_data_pts,centroids), axis=1)
#     raise NotImplementedError('kmeans_quantize function not implemented.')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return indices


def get_bags_of_sifts(image_arrays, vocabulary, step_size = 10):

    # load vocabulary
    vocab = vocabulary

    vocab_size = len(vocab)
    num_images = len(image_arrays)

    feats = np.zeros((num_images, vocab_size))

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    for i in range(len(image_arrays)):
        image = image_arrays[i]
        width = image.shape[1]
        height = image.shape[0]
        interest_x = np.arange(10, width - 10, step_size)
        interest_y = np.arange(10, height - 10, step_size)
        interest = np.transpose([np.tile(interest_x, interest_y.shape[0]), np.repeat(interest_y, interest_x.shape[0])])
        x = interest[:,0]
        y = interest[:,1]

        image = torch.from_numpy(np.array(image, dtype=np.float32))
        img_tensor = image.reshape((1, 1, image.shape[0], image.shape[1]))
        sift_features = get_siftnet_features(img_tensor, x, y)
        
        histogram = np.histogram(kmeans_quantize(sift_features, vocab), bins=np.arange(vocab_size+1))
        norm = np.linalg.norm(histogram[0])   

        if norm == 0:
                feats[i] = histogram[0]
        else:
                feats[i] = histogram[0]/norm
#     raise NotImplementedError('get_bags_of_sifts function not implemented.')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return feats
