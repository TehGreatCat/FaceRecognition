import numpy
import itertools
Feature_types = [x for x in range(10)]


class ModFeatures(object):

    def __init__(self, a, b, feature_type):
        if a > b:
            a, b = b, a
        self.type = feature_type
        self.a = a
        self.b = b
        self.dist = 0

    def get_score(self, img):
        if not(isinstance(img, list) or isinstance(img, numpy.ndarray)):
            image = img.img
        elif isinstance(img, numpy.ndarray):
            print(type(img))
            image = img.flatten()
        else:
            image = list(itertools.chain.from_iterable(img))
        self.dist = image[self.a] - image[self.b]
        if self.type == 0:
            return 1 if self.dist < 0 else 0
        if self.type == 1:
            return 1 if abs(self.dist) < 5 else 0
        if self.type == 2:
            return 1 if abs(self.dist) < 10 else 0
        if self.type == 3:
            return 1 if abs(self.dist) < 25 else 0
        if self.type == 4:
            return 1 if abs(self.dist) < 50 else 0
        if self.type == 5:
            return 1 if self.dist >= 0 else 0
        if self.type == 6:
            return 1 if abs(self.dist) >= 5 else 0
        if self.type == 7:
            return 1 if abs(self.dist) >= 10 else 0
        if self.type == 8:
            return 1 if abs(self.dist) >= 25 else 0
        if self.type == 9:
            return 1 if abs(self.dist) >= 50 else 0
