from __future__ import division
from numpy import *
from Mod_classifier_func import Feature_types, ModFeatures


class Im:

        def __init__(self, label, weight, img):
            self.label = label
            self.weight = weight
            self.img = img


class AdaBoost:

    def __init__(self, set_pos, set_neg, T):
        self.set_pos = set_pos
        self.set_neg = set_neg
        self.num_pos = len(self.set_pos)
        self.num_neg = len(self.set_neg)
        self.weights_pos = ones(self.num_pos)/(2 * self.num_pos)
        self.weights_pos = ones(self.num_neg)/(2 * self.num_neg)
        self.T = T

    def train_simple(self):  # на вход должны приходить экз-ры класса
        for p in self.set_pos:
            p.weight = 1 / (2 * self.num_pos)
            p.label = 1
        for n in self.set_neg:
            n.weight = 1 / (2 * self.num_neg)
            n.label = 0

        features = []
        for f in Feature_types:
            for a in range(0, 25, 1):
                for b in range(0, 25, 1):
                    features.append(ModFeatures(a, b, f))   # Все features из списка

        images = self.set_pos + self.set_neg
        # print(images[0])

        votes = dict()
        i = 0
        for feature in features:
            feature_votes = array(list(map(lambda img: [img, feature.get_score(img)], images)))  # несовпадение классов,
            # в ModClassifier приходит херня, а не чистое изображение.
            votes[feature] = feature_votes  # список списков [изобр, {-1, 1} для  изобр при данном feature]
            i += 1
            if i % 1000 == 0:
                print('hey b0ss')
                break

        # select classifiers

        classifiers = []
        used = []

        for i in range(self.T):
            print("Stage number ", i)
            classification_errors = dict()
            # Нормализация весов
            norm_factor = 1 / sum(list(map(lambda im: im.weight, images)))
            for image in images:
                image.weight *= norm_factor

            for feature, feature_votes in iter(votes.items()):
                if feature in used:
                    continue
                # print(list(feature_votes))
                error = sum(list(map(lambda im, vote: im.weight if im.label != vote else 0,
                                feature_votes[:, 0], feature_votes[:, 1])))
                # Рассчет ошибки для каждого feature
                # if error >= 0.5:
                #       continue
                classification_errors[error] = feature  # не наоборот? nope

            # выбор лучшего слабого классификатора
            errors = list(classification_errors.keys())
            # print(errors)
            best_err = errors[argmin(errors)]
            # print(best_err)  # получается 0.5, что при подсчете альфа дает ноль, и алгоритм заканчивается
            # ошибка в подсчетах? откуда берется 0.5?
            feature = classification_errors[best_err]
            used.append(feature)
            feature_weight = log((1 - best_err)/best_err)  # это alpha в strong clf
            print(feature_weight)
            classifiers.append((feature, feature_weight))  # 0.5 * log((1 - best_err)/best_err)

            best_feature_votes = votes[feature]
            for f_vote in best_feature_votes:
                im = f_vote[0]
                vote = f_vote[1]
                if im.label != vote:
                    pass
                else:
                    im.weight *= best_err/(1 - best_err)
            print("Stage done, next one")
            print("--------------------")
            if i % 20 == 0:
                print(classifiers)

        return classifiers   # слабые


class StrongClassifier(object):

    def __init__(self, clf_list, img):
        self.clf_list = array(clf_list)[:, 0]
        self.alpha = array(clf_list)[:, 1]
        self.img = img

    def get_result(self):
        left = sum(list(map(lambda alpha, clf: alpha * clf.get_score(self.img), self.alpha, self.clf_list)))
        right = 0.5 * sum(self.alpha)  # взять из статьи про пол <<<<<
        return 1 if left >= right else 0
