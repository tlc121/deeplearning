import Integrated as ii
import numpy as np

FeatureTypes = ['TWO_HORIZONAL', 'THREE_VERTICAL']

class HaarFeature(object):

    def __init__(self, feature_type, pos, feature_width, feature_height, threshold):
        self.type = feature_type
        self.position = pos
        self.width = feature_width
        self.height = feature_height
        self.threshold = threshold
        self.weight = 1


    def get_score(self, img):
        if self.type is 'TWO_HORIZONAL':
            return self.two_horizonal(img)
        elif self.type is 'THREE_VERTICAL':
            return self.three_vertical(img)


    def two_horizonal(self, img):
        #calculate the up block
        left_top_black = self.position
        right_bottom_black = [self.position[0] + self.height/2, self.position[1] + self.width]
        feature_val_black = ii.sum_region(img, left_top_black, right_bottom_black)

        #calculate the bottom block
        left_top_white = [self.position[0] + self.height/2, self.position[1]]
        right_bottom_white = [self.position[0] + self.height, self.position[1] + self.width]
        feature_val_white = ii.sum_region(img, left_top_white, right_bottom_white)

        #return difference
        return np.abs(feature_val_black - feature_val_white)

    def three_vertical(self, img):
        left_top_left = self.position
        right_bottom_left = [self.position[0] + self.height, self.position[1] + self.width/3]
        feature_val_left = ii.sum_region(img, left_top_left, right_bottom_left)

        left_top_middle = [self.position[0], self.position[1] + self.width/3]
        right_bottom_middle = [self.position[0] + self.height, self.position[1] + (2*self.width)/3]
        feature_val_middle = ii.sum_region(img, left_top_middle, right_bottom_middle)

        left_top_right = [self.position[0], self.position[1] + self.width/3]
        right_bottom_right = [self.position[0] + self.height, self.position[1] + self.width]
        feature_val_right = ii.sum_region(img, left_top_right, right_bottom_right)

        return np.abs(feature_val_left + feature_val_right - feature_val_middle)


    def get_vote(self, image):
        value = self.get_score(image)
        return 1 * self.weight if value > self.threshold else -1 * self.weight
