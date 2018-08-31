import cv2
import HaarFeature
import numpy as np
import os
import Integrated
import Boosting
import test
from tqdm import tqdm

def preprocess(img):
    '''
    this part will preprocess the input image:
    1. normalize the image
    2. reshape it into 24*24
    '''
    im_double = im2double(img)
    result = cv2.resize(im_double, (24, 24), cv2.INTER_CUBIC)
    return result

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

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float32') - min_val) / (max_val - min_val)
    return out

def loading_negative_img(negative_path):
    iis_negative = []
    folderlist = os.listdir(negative_path)

    for filename in tqdm(os.listdir(negative_path)):
        if filename.endswith('.pgm') or filename.endswith('.jpg'):
            image = cv2.imread(negative_path + filename, cv2.IMREAD_GRAYSCALE)
            img_pro = preprocess(image)
            iis = Integrated.IntegratedImage(img_pro)
            iis_negative.append(img2vector(iis))
    return iis_negative

def loading_positive_img(positive_path):
    iis_positive = []
    for filename in tqdm(os.listdir(positive_path)):
        if filename.endswith('.pgm') or filename.endswith('.jpg'):
            image = cv2.imread(positive_path + filename, cv2.IMREAD_GRAYSCALE)
            img_pro = preprocess(image)
            iis = Integrated.IntegratedImage(img_pro)
            iis_positive.append(img2vector(iis))
    return iis_positive

def accuracy(test_path_truep, test_path_truen, classifiers):
    folderlist_truep = os.listdir(test_path_truep)
    folderlist_truen = os.listdir(test_path_truen)
    corr = 0
    total = 0
    print 'now test the accuracy!'
    for folder in tqdm(folderlist_truep):
        if os.path.isdir(test_path_truep + folder):
            for filename in os.listdir(test_path_truep + folder):
                if filename.endswith('.pgm') or filename.endswith('.jpg'):
                    total += 1
                    image = cv2.imread(test_path_truep + folder + '/' + filename, cv2.IMREAD_GRAYSCALE)
                    img_pro = preprocess(image)
                    iis = Integrated.IntegratedImage(img_pro)
                    if test.testify(iis, classifiers) is 1:
                        corr += 1
    truep = float(corr)/float(total)

    corr = 0
    total = 0
    for folder in tqdm(folderlist_truen):
        if os.path.isdir(test_path_truen + folder):
            for filename in os.listdir(test_path_truen + folder):
                if filename.endswith('.pgm') or filename.endswith('.jpg'):
                    total += 1
                    image = cv2.imread(test_path_truen + folder + '/' + filename, cv2.IMREAD_GRAYSCALE)
                    img_pro = preprocess(image)
                    iis = Integrated.IntegratedImage(img_pro)
                    if test.testify(iis, classifiers) is -1:
                        corr += 1
    truen = float(corr)/float(total)
    return truep, truen

def train():
    positivepath = '/Users/xavier0121/Desktop/positive/allfaces/'
    negativepath = '/Users/xavier0121/Desktop/negative/allfaces/'
    iis_positive = loading_positive_img(positivepath)
    iis_negative = loading_negative_img(negativepath)
    test_path_truep = '/Users/xavier0121/Desktop/Test/'
    test_path_truen = '/Users/xavier0121/Desktop/Test1/'
    num_classifiers = 1500
    min_feature_height = 8
    min_feature_width = 8
    max_feature_height = 10
    max_feature_width = 10
    classifiers = Boosting.learn(iis_positive, iis_negative, num_classifiers, min_feature_width, max_feature_width, min_feature_height, max_feature_height)
    truep, falsen = accuracy(test_path_truep, test_path_truen, classifiers)
    print 'The true positive rate: ' + str(truep)
    print 'The true negative rate: ' + str(falsen)
    return classifiers
