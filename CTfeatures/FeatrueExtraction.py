import numpy as np
import cv2
import os
from sklearn import cluster
from sklearn import neighbors
from tqdm import tqdm

class gv:
    num_clusters = 10

def featureExtract(tuci_path, no_path):
    allFeatures = []
    features_tuci = []
    features_no = []
    features = {'tuci':[], 'no':[]}
    histogram_tuci = np.zeros(shape=[1, gv.num_clusters])
    histogram_no = np.zeros(shape=[1, gv.num_clusters])


    for filename in tqdm(os.listdir(tuci_path)):
        image = cv2.imread(tuci_path + filename, cv2.IMREAD_GRAYSCALE)
        num_layers, row, col = image.shape
        #get the each layer
        for i in range(0, num_layers):
            #get the layer
            image_temp = image[i:i+1, :]
            #using SIFT to get the all features
            sift =  cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(image_temp, None)
            num_features, dim = des.shape
            for j in range(0, num_features):
                features_tuci.append(des[j:j+1, :])
                allFeatures.append(des[j:j+1, :])
    features['tuci'] = features_tuci
    num_tuci_features = len(features_tuci)


    for filename in tqdm(os.listdir(no_path)):
        image = cv2.imread(no_path + filename, cv2.IMREAD_GRAYSCALE)
        num_layers, row, col = image.shape
        # get the each layer
        for i in range(0, num_layers):
            # get the layer
            image_temp = image[i:i + 1, :]
            # using SIFT to get the all features
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(image_temp, None)
            num_features, dim = des.shape
            for j in range(0, num_features):
                features_no.append(des[j:j+1, :])
                allFeatures.append(des[j:j+1, :])
    features['no'] = features_no
    num_no_features = len(features_no)

    #     if filename.endswith('jpg') or filename.endswith('pgm'):
    #         image = cv2.imread(no_path + filename, cv2.IMREAD_GRAYSCALE)
    #         # using SIFT to get the all features
    #         sift = cv2.xfeatures2d.SIFT_create()
    #         kp, des = sift.detectAndCompute(image, None)
    #         num_features, dim = des.shape
    #         for j in range(0, num_features):
    #             features_no.append(des[j, :])
    #             allFeatures.append(des[j, :])
    #         features['no'] = features_tuci
    # print len(allFeatures)

    #using kmeans to cluster the features
    print 'Do Kmeans'
    estimator = cluster.KMeans(n_clusters = gv.num_clusters, random_state=0).fit(allFeatures)
    labels = estimator.labels_
    print 'Kmeans Done!'

    #create the histogram for tuci
    count = 0
    for feature in features_tuci:
        idx = labels[count]
        histogram_tuci[0, idx] +=  1
        count += 1

    for feature in features_no:
        idx = labels[count]
        histogram_no[0, idx] += 1
        count += 1

    #normalize the histogram
    histogram_tuci = histogram_tuci / float(sum(histogram_tuci[0]))
    histogram_no = histogram_no / float(sum(histogram_no[0]))
    print 'Done!'
    return histogram_tuci, histogram_no, labels, allFeatures






def test_accuracy(testpath_tuci, testpath_no, labels, allFeatures, histogram_tuci, histogram_no):
    total = 0
    correct = 0
    #test the tuci first
    for filename in os.listdir(testpath_tuci):
        total += 1
        image = cv2.imread(testpath_tuci + filename, cv2.IMREAD_GRAYSCALE)
        num_layers, row, col = image.shape
        histogram_temp = np.zeros(shape=[1, gv.num_clusters])
        for i in range(0, num_layers):
            # get the layer
            image_temp = image[i:i+1, :]

            # using SIFT to get the all features
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(image_temp, None)
            num_features, dim = des.shape

            # using KNN to classify the feature
            clf = neighbors.NearestCentroid(metric='euclidean', shrink_threshold=None)
            clf.fit(allFeatures, labels)
            for j in range(0, num_features):
                feature_temp = des[j:j+1, :]
                idx = clf.predict(feature_temp)
                histogram_temp[0, idx] += 1
            histogram_temp = histogram_temp/float(sum(histogram_temp[0]))
            if np.linalg.norm(histogram_temp - histogram_tuci) < np.linalg.norm(histogram_temp - histogram_no):
                correct += 1

    # then test the no_tuci
    for filename in os.listdir(testpath_no):
        total += 1
        image = cv2.imread(testpath_no + filename, cv2.IMREAD_GRAYSCALE)
        num_layers, row, col = image.shape
        histogram_temp = np.zeros(shape=[1, gv.num_clusters])
        for i in range(0, num_layers):
            # get the layer
            image_temp = image[i:i + 1, :]

            # using SIFT to get the all features
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(image_temp, None)
            num_features, dim = des.shape

            # using KNN to classify the feature
            clf = neighbors.NearestCentroid(metric='euclidean', shrink_threshold=None)
            clf.fit(allFeatures, labels)
            for j in range(0, num_features):
                feature_temp = des[j:j+1, :]
                idx = clf.predict(feature_temp)
                histogram_temp[0, idx] += 1
            histogram_temp = histogram_temp/float(sum(histogram_temp[0]))
            if np.linalg.norm(histogram_temp - histogram_tuci) > np.linalg.norm(histogram_temp - histogram_no):
                correct += 1

    print float(correct)/float(total)

if __name__ == '__main__':
    tuci_path = '/Users/xavier0121/Desktop/positive1/'
    no_path = '/Users/xavier0121/Desktop/negative1/'
    histogram_tuci, histogram_no, labels, allFeatures = featureExtract(tuci_path, no_path)
    test_accuracy(tuci_path, no_path, labels, allFeatures, histogram_tuci, histogram_no)


