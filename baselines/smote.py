"""
Codes of paper 'SMOTE: Synthetic Minority Over-sampling Technique'

author: Dongyang Yan
github: github.com/fiyen
last modification date: 2021/11/08
"""
import random
import numpy as np


def multiple_xy(x, y, cut_num):
    """
    multiple the number of x and y accordingly.
    :param x:
    :param y:
    :return:
    """
    sum_classes = np.sum(y, axis=0)
    num_classes = sum_classes.shape[0]
    max_num = int(np.max(sum_classes))
    sort_data = {i: {'data': [], 'labels': []} for i in range(num_classes)}
    for index, label in enumerate(y):
        i = np.argmax(label)
        sort_data[i]['data'].append(x[index])
        sort_data[i]['labels'].append(label)
    for i, num in enumerate(sum_classes):
        num = int(num)
        r = max_num % num
        n = max_num // num
        if cut_num < num:
            r = 0
            n = 1
        else:
            r = cut_num % num
            n = cut_num // num
        indexes = [x for x in range(num)]
        sample_indexes = random.sample(indexes, r)
        r_data = [sort_data[i]['data'][x] for x in sample_indexes]
        r_labels = [sort_data[i]['labels'][x] for x in sample_indexes]
        sort_data[i]['data'] = multiple_obj(sort_data[i]['data'], n) + r_data
        sort_data[i]['labels'] = multiple_obj(sort_data[i]['labels'], n) + r_labels
        new_x = []
        new_labels = []
        for i in range(num_classes):
            new_x += sort_data[i]['data']
            new_labels += sort_data[i]['labels']
        # shuffles
        new_x = np.array(new_x)
        new_labels = np.array(new_labels)
        index_new = [i for i in range(new_x.shape[0])]
        random.shuffle(index_new)
        x = np.array([new_x[i] for i in index_new])
        y = np.array([new_labels[i] for i in index_new])
    return x, y


def multiple_obj(objs, n=0):
    """
    objs中的元素倍增后返回
    :param n:
    :param objs: np.array, list
    :return:
    """

    if not isinstance(objs, (list, np.ndarray)):
        raise AttributeError('Only support {list, ndarray} type. objs" type is ', type(objs))
    if not isinstance(objs, list):
        objs = objs.tolist()
    out = []
    for i in range(n):
        out = out + objs
    return out


class SMOTE:
    """
    smote algorithm
    N: Amount of SMOTE N%
    k: Number of nearest neighbors k
    Note: the input and output are all lists.
    """
    def __init__(self, k=5):
        self.k = k

    def oversample(self, minority, N):
        """
        :param minority: samples of minority class, list
        :param N: Amount of SMOTE N%
        :return: (N/100) * T synthetic minority class samples
        """
        new_samples = []
        T = len(minority)
        if N < 100:
            T = int((N / 100) * T)
            N = 100
        N = int(N / 100)
        T = random.sample(minority, T)
        for i in range(len(T)):
            smote_samples = self.populate(N, i, T)
            new_samples += smote_samples
        return new_samples

    def populate(self, N, i, minority):
        """
        Function to generate the synthetic samples
        :param N: the times of expansion
        :param i: original data i
        :param minority:
        :return:
        """
        new_samples = []
        nnarray = self.get_knn_indices(i, minority)
        while N > 0:
            ind = random.sample(nnarray, 1)[0]
            dif = np.array(minority[ind]) - np.array(minority[i])
            gap = np.array([random.randint(0, 1) for i in range(len(minority[0]))])
            new_sample = (np.array(minority[i]) + gap * dif).tolist()
            new_samples.append(new_sample)
            N -= 1
        return new_samples

    def oversample_v2(self, minority, cut_num):
        """
        increase the sample size to 'cut_num'
        :param cut_num:
        :param minority:
        :return:
        Note: if cut_num <= the size of minority, return minority itself directly
        """
        T = len(minority)
        if cut_num <= T:
            return minority
        else:
            samples = [v for v in minority]
            n = int(cut_num // T)
            r = int(cut_num % T)
            if n > 1:
                samples += self.oversample(minority, (n - 1) * 100)
            samples += self.oversample(minority, r / T * 100)
        return samples

    def data_expand(self, data, label, cut_num=0):
        """
        To expand the minority classes
        :param cut_num: the amount of every minority class will be expanded less than cut_num
        :param data: features vectors
        :param label: one hot vectors
        :return: np.array
        """
        class_num = len(list(label[0]))
        sorted_data = {x: {'data': [], 'label': []} for x in range(class_num)}
        for ind, d in enumerate(data):
            sorted_data[np.argmax(label[ind])]['data'].append(d)
            sorted_data[np.argmax(label[ind])]['label'] = label[ind]
        for i in range(class_num):
            sorted_data[i]['data'] = self.oversample_v2(sorted_data[i]['data'], cut_num)
        new_data = []
        new_label = []
        for i in range(class_num):
            label = sorted_data[i]['label']
            for line in sorted_data[i]['data']:
                new_data.append(line)
                new_label.append(label)
        len_data = len(new_data)
        new_data = np.array(new_data)
        new_label = np.array(new_label)
        indices = np.random.permutation(range(len_data))
        new_data = new_data[indices]
        new_label = new_label[indices]
        return new_data, new_label

    def get_knn_indices(self, i, minority):
        """
        get the indices of k nearest neighbors vectors of i in minority
        :param i:
        :param minority:
        :return: indices, list
        """
        if len(minority) < self.k:
            k = len(minority)
        else:
            k = self.k
        feature_i = np.array(minority[i])
        min_array = np.array(minority)
        cos = np.dot(min_array, feature_i) / np.sqrt(np.sum(min_array * min_array, axis=-1)
                                                     * np.sum(feature_i * feature_i) + 1e-9)
        flat = cos.flatten()
        indices = np.argpartition(flat, -k)[-k:]
        indices = indices[np.argsort(-flat[indices])]
        return [x for x in indices]


if __name__ == '__main__':
    a = [[1, 2, 3], [1, 3, 4], [2, 4, 5], [2, 3, 4], [1, 2, 4], [1, 3, 5], [2, 4, 6]]
    b = [[0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0]]
    se = SMOTE(5)
    test = se.oversample(a, 200)
    test_1 = se.oversample_v2(a, 10)
    test_2 = se.data_expand(a, b, cut_num=10)