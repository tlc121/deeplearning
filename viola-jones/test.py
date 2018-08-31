import numpy as np
import Boosting
import util
import Integrated
import cv2
import HaarFeature
from tqdm import tqdm


def sgn(value):
    if value > 0:
        return 1
    else:
        return -1

def testify(img, classifiers):
    value = 0
    for feature in classifiers:
        temp = feature.get_vote(img)
        value += temp
    return sgn(value)

def detectfaces(img, classifiers):
    row1, col1, dimension = img.shape
    image_resize = cv2.resize(img, (col1/2, row1/2), cv2.INTER_CUBIC)
    image = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
    min_width = 50
    max_width = 100
    for size in tqdm(range(min_width, max_width, 10)):
        for i in range(0, row1 - size, 10):
            for j in range(0, col1 - size, 10):
                img_crop = image[i:i+size, j:j+size]
                img_pro = util.preprocess(img_crop)
                iis = Integrated.IntegratedImage(img_pro)
                if testify(iis, classifiers) is 1:
                    cv2.rectangle(image_resize, (j, i), (j + size, i + size), (0, 255, 0), 2)
    cv2.imshow('Face Detector', image_resize)
    cv2.waitKey(50000)
    cv2.destroyAllWindows()

