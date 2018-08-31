import cv2
import HaarFeature
import numpy as np
import os
import Integrated
from tqdm import tqdm

def img2vector(img):
    row, col = img.shape
    result = np.zeros(shape=[row*col, 1])
    for i in range(0, row):
        for j in range(0, col):
            result[i*col + j, 0] = img[i, j]
    return result

def vector2image(vec):
    row = 24
    col = 24
    res = np.zeros(shape=[row, col])
    for i in range (0,row):
        for j in range(0, col):
            res[i, j] = vec[i*col + j]
    return res

def get_feature_vote(feature, img):
    return feature.get_vote(img)

def createfeatures(min_feature_width, max_feature_width, min_feature_height, max_feature_height):
    img_height, img_width = 24, 24
    #create features
    features = []
    for featuretype in HaarFeature.FeatureTypes:
        for feature_width in range (min_feature_width, max_feature_width):
            for feature_height in range (min_feature_height, max_feature_height):
                for x in range(0, img_height - feature_height):
                    for y in range(0, img_width - feature_width):
                        pos = [x, y]
                        feature = HaarFeature.HaarFeature(featuretype, pos, feature_width, feature_height, 3.45)
                        features.append(feature)
    return features

def learn(positive_iis, negative_iis, num_classifiers, min_feature_width, max_feature_width, min_feature_height, max_feature_height):
    #get the size and number of the image
    num_positive = len(positive_iis)
    num_negative = len(negative_iis)
    num_images = num_negative + num_positive
    img_height, img_width = positive_iis[0].shape


    #create initial weights and labels
    pos_weights = np.ones(num_positive) * 1./(num_positive*2)
    neg_weights = np.ones(num_negative) * 1./(num_negative*2)
    weights = np.hstack((pos_weights, neg_weights))
    labels = np.hstack((np.ones(num_positive), np.ones(num_negative) * -1))
    images = positive_iis + negative_iis


    #create features for all sizes and locations
    features = createfeatures(min_feature_width, max_feature_width, min_feature_height, max_feature_height)
    num_features = len(features)

    # label index of features
    feature_index = list(range(num_features))

    #create a matrix to store the vote
    votes = np.zeros((num_images, num_features))
    for i in tqdm(range(num_images)):
        image = vector2image(images[i])
        for j in range(num_features):
            features_temp = features[j]
            result = get_feature_vote(features_temp, image)
            votes[i, j] = result
    print 'votes done!'


    #select classifers
    classifier = []

    for i in tqdm(range(num_classifiers)):
        classification_error = np.zeros(len(feature_index))

        weights *= 1./np.sum(weights)

        for f in range(len(feature_index)):
            f_idx = feature_index[f]
            error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0, range(num_images)))
            classification_error[f] = error

        min_error_idx = np.argmin(classification_error)
        best_error = classification_error[min_error_idx]
        best_feature_idx = feature_index[min_error_idx]

        #set feature weight
        best_feature = features[best_feature_idx]
        feature_weights = 0.5 * np.log((1 - best_error) / best_error )
        best_feature.weight = feature_weights
        classifier.append(best_feature)

        #update the weights
        weights = np.array(list(map(lambda img_idx: weights[img_idx]*np.sqrt((1-best_error)/best_error) if labels[img_idx] != votes[img_idx, best_feature_idx] else weights[img_idx]*np.sqrt(best_error/(1-best_error)), range(num_images))))
        feature_index.remove(best_feature_idx)

    print 'training done!'
    return classifier




