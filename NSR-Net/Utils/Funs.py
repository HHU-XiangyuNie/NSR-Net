import torch
import random
import math
import numpy as np
from sklearn import metrics
from operator import truediv

"""
This function is used to randomly extract training samples.
"""
def spilt_train(all_label, rate=0.1, rand_index = False, need_index=False):
    [n1, n2] = all_label.shape
    train_label = torch.zeros([n1, n2])
    del_train_label = all_label
    num_classes = int(torch.max(all_label))
    train_index = []
    for i in range(1, num_classes + 1):
        i_class_index = torch.nonzero(all_label == i)
        i_class_index_num = len(i_class_index)
        i_shuffle_list = list(range(i_class_index_num))
        if rand_index:
            random.seed(rand_index)
        random.shuffle(i_shuffle_list)
        if rate < 1:
            i_class_training_num = math.ceil(i_class_index_num * rate)
            if i_class_training_num == 1:
                i_class_training_num += 1
        else:
            if rate >= 0.5*i_class_index_num:
                i_class_training_num = math.ceil(i_class_index_num * 0.5)
            else:
                i_class_training_num = rate

        temp = i_shuffle_list[0:i_class_training_num]
        i_class_train_index = i_class_index[temp]
        i_class_train_index = [T.tolist() for T in i_class_train_index]
        train_index.append(i_class_train_index)
    train_index = sum(train_index, [])
    train_index = torch.tensor(train_index)
    for each_index in train_index:
        train_label[each_index[0], each_index[1]] = all_label[each_index[0], each_index[1]]

    del_train_label [train_index[:,0], train_index[:,1]] = 0

    if need_index:
        return train_label, del_train_label, train_index
    else:
        return train_label, train_index


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return average_acc, each_acc,
"""
This function is used to evaluate the classification accuracy.
"""
def kappa_statistic(GT, Predict):
    confusion_mat = metrics.confusion_matrix(GT, Predict)
    OA = metrics.accuracy_score(GT, Predict)
    kappa = metrics.cohen_kappa_score(GT, Predict)
    AA, class_AA = AA_andEachClassAccuracy(confusion_mat)

    results = {'OA': OA,
               'AA': AA,
               'class_AA': class_AA,
               'kappa': kappa}
    return results

"""
This function is used to color the classification map
"""
def color_picture(class_map, num_class):
    color_table = [[0, 168, 132],
                    [76, 0, 115],
                    [0,   0, 0],
                    [190, 255,  232],
                    [255, 0, 0],
                    [115, 0,  0],
                    [205, 205, 102],
                    [137, 90,  68],
                    [215, 158, 158],
                    [255, 115, 223],
                    [0, 0, 255],
                    [156, 156, 156],
                    [115,   223,  255],
                    [0,255, 0],
                    [255, 255, 0],
                    [255, 170, 0]]

    [n1, n2] = class_map.shape
    class_map = list(class_map.flatten())
    for index, label in enumerate(class_map):
        for cur_class in range(num_class):
            if cur_class+1 == label:
                class_map[index] = color_table[cur_class]
    class_map = np.array(class_map)
    class_map = class_map.reshape((n1, n2, -1))
    return class_map


