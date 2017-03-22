from AdaBoost import *
import cv2


def create_image_list(folder_pos, folder_neg, num_of_pos, num_of_neg):

    pos_image_list = []
    neg_image_list = []

    # print("here", image)

    for i in range(1, num_of_pos):
        img = Im(1, 1, hstack(cv2.resize(cv2.imread(folder_pos + "\\" + str(i) + ".pgm", 0), (24, 24))))
        pos_image_list.append(img)

    for j in range(num_of_neg):
        img = Im(0, 1, hstack(cv2.resize(cv2.imread(folder_neg + "\\" + str(j) + ".bmp", 0), (24, 24))))
        neg_image_list.append(img)

    # print(pos_image_list[0].img)
    return [pos_image_list, neg_image_list]


def pgm_simplifier(image):
    image_new = [[] for x in range(len(image))]
    for i in range(0, len(image) - 1):
        for j in range(0, len(image[i]) - 1):
            image_new[i].append(image[i][j].tolist().pop())
    return image_new


def main(number_of_stages, set_pos, set_neg):
    trainer = AdaBoost(set_pos, set_neg, number_of_stages)
    classifiers = trainer.train_simple()
    image = cv2.imread("1.pgm")
    print(StrongClassifier(classifiers, pgm_simplifier(image)).get_result())
    # print(classifiers)

images = create_image_list("FinalGood", "Bad", 400, 450)
main(3, images[0], images[1])



