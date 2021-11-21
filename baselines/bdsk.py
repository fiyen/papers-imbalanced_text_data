"""
codes of paper
A Bi-directional Sampling based on K-Means Method for Imbalance Text Classification
Jia Song, Xianglin Huang, Sijun Qin, Qing Song

author: Dongyang Yan
github: github.com/fiyen
last modification date: 2021/11/08
"""
from sklearn.cluster import KMeans
import numpy as np
from .smote import SMOTE


class BDSK:
    def fit(self, data, label):
        """
        求出数据集各类样本数的平均值
        :param label: 每个data样本对应的标记
        :param data: 行代表样本，列代表特征向量的每个维度
        :return: 处理后的样本
        """
        class_num = len(list(label[0]))
        sorted_data = {x: {'data': [], 'label': []} for x in range(class_num)}
        for ind, d in enumerate(data):
            sorted_data[np.argmax(label[ind])]['data'].append(d)
            sorted_data[np.argmax(label[ind])]['label'] = label[ind]
        num = 0
        for lab in sorted_data.keys():
            num += len(sorted_data[lab]['data'])
        self.k = int(num / len(sorted_data))
        return sorted_data

    def under_sampling(self, majority, k_neighbor=1, k=None):
        """
        对majority class降采样
        :param k_neighbor: 选取聚类中心周围节点数量
        :param majority:
        :param k: int
        :return: array like (k, num_features)
        """
        if k is None:
            k = self.k
        if np.ndim(majority) != 2:
            raise ValueError("The rank of majority should be 2 but now is %d. " % np.ndim(majority))
        if len(list(majority)) < k:
            print("\033[32;0mWarning: The number of samples of majority is less than k.\033[0m")
            return majority
        kmeans = KMeans(n_clusters=k)
        majority = np.array(majority)
        kmeans.fit(majority)
        k_maj = kmeans.cluster_centers_
        return np.concatenate([self.get_nearest_neighbor(center, majority, k_neighbor) for center in k_maj], axis=0)

    def get_nearest_neighbor(self, center, arrays, k):
        center = np.reshape(center, (-1, center.size))
        dis = np.sum(np.power(arrays - center, 2), axis=-1)
        flat = dis.flatten()
        indices = np.argpartition(flat, -k)[-k:]
        indices = indices[np.argsort(-flat[indices])]
        return arrays[indices]

    def over_sampling(self, minority, s=2, k=None):
        """
        对minority超采样至超过k
        :param s: 每次迭代用smote产生的样本数量
        :param minority:
        :param k:
        :return: array like (~k, num_features)
        """
        if k is None:
            k = self.k
        if np.ndim(minority) != 2:
            raise ValueError("The rank of minority should be 2 but now is %d. " % np.ndim(minority))
        if len(list(minority)) > k:
            print("\033[32;0mWarning: The number of samples of minority is larger than k.\033[0m")
            return minority
        smote = SMOTE()
        minority = list(minority)
        if len(list(minority)) < 5:
            minority = smote.oversample_v2(minority, min(5, k))
        while True:
            min_1, min_2 = self.k_partition(minority, 2)
            num_1 = min_1.shape[0]
            num_2 = min_2.shape[0]
            if num_1 * num_2 > 0:
                if num_1 > num_2:
                    min_2 = smote.oversample_v2(list(min_2), num_2 + s)
                else:
                    min_1 = smote.oversample_v2(list(min_1), num_1 + s)
            else:
                if num_1 > 0:
                    min_1 = smote.oversample_v2(list(min_1), num_1 + s)
                else:
                    min_2 = smote.oversample_v2(list(min_2), num_2 + s)
            minority = list(min_1) + list(min_2)
            if len(minority) >= k:
                return np.array(minority)

    def k_partition(self, data, k):
        """
        把data用kmeans方法分成k分，
        :param data: array or list like (num_samples, num_features)
        :return: list, k parts
        """
        data = np.array(data)
        parts = KMeans(k).fit_predict(data)
        return [data[parts == i] for i in range(k)]

    def expand_data(self, fit_output, cut_num=None, k_neighbor=None):
        """
        expand the datasets
        :param k_neighbor: 降采样中需要考虑周围节点数量
        :param cut_num:
        :param fit_output: the output of fit
        :return:
        """
        for key, val in fit_output.items():
            data = val['data']
            print('key', key)
            print('len', len(data))
            if cut_num is None:
                if len(data) > self.k:
                    data = list(self.under_sampling(data))
                elif len(data) < self.k:
                    data = list(self.over_sampling(data))
                else:
                    data = data
            else:
                if len(data) < cut_num:
                    data = list(self.over_sampling(data, k=cut_num))
                else:
                    if k_neighbor is not None:
                        data = list(self.under_sampling(data, k_neighbor))
                    else:
                        data = data
            print('new_len', len(data))
            val['data'] = data
            fit_output[key] = val
        new_data = []
        new_label = []
        for i in fit_output.keys():
            label = fit_output[i]['label']
            for line in fit_output[i]['data']:
                new_data.append(line)
                new_label.append(label)
        len_data = len(new_data)
        new_data = np.array(new_data)
        new_label = np.array(new_label)
        indices = np.random.permutation(range(len_data))
        new_data = new_data[indices]
        new_label = new_label[indices]
        return new_data, new_label


if __name__ == '__main__':
    a = [[1, 2, 3], [1, 3, 4], [2, 4, 5], [2, 3, 4], [1, 2, 4], [1, 3, 5], [2, 4, 6]]
    b = [[0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]]
    bdsk = BDSK()
    data = bdsk.fit(a, b)
    new_data, new_label = bdsk.expand_data(data)
