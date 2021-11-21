"""
codes of paper 'Imbalanced Text Categorization Based on Positive and Negative Term Weighting Approach'

author: Dongyang Yan
github: github.com/fiyen
last modification date: 2021/11/08
"""
import numpy as np


class PNF:
    def __init__(self):
        self.name = 'PNF'

    def fit(self, data, label):
        """
        PNF2 = (P(ti|cj) - P(ti|~Cj)) / (P(ti|Cj) + P(ti|~Cj))
        P(ti|Cj) = a / (a + b)  P(ti|~Cj) = c / (c + d)
        PNF2 = (ad - bc) / (2ac + ad + bc)
        :param data: features vectors
        :param label: one hot vectors
        """
        sorted_data = category(data, label)
        cross_tf = abcd(sorted_data)
        cross_tf = np.transpose(cross_tf, axes=[2, 0, 1])
        a, b, c, d = cross_tf[0], cross_tf[1], cross_tf[2], cross_tf[3]
        PNF = (a * d - b * c) / (2 * a * c + a * d + b * c + 1e-05)
        self.PNF = PNF

    def pnf(self, data, label, ptype='2', shuffle=True):
        """
        :param shuffle: whether or not to shuffle the data
        :param data: features vectors
        :param label: one hot vectors
        :param ptype: '1' will return PNF1 and '2' will return PNF2, PNF1 = 1 + PNF2
        :return: an array with the same shape of data and label
        """
        sorted_data = category(data, label)
        if ptype == '2':
            PNF = self.PNF
        elif ptype == '1':
            PNF = 1 + self.PNF
        else:
            raise ValueError('ptype ERROR!')
        data = np.array(data)
        out_data = []
        out_label = []
        if not shuffle:
            for ind, x in enumerate(data):
                data[ind] = x * PNF[np.argmax(label[ind])]
            out_data = data
            out_label = label
        else:
            for i in range(len(sorted_data)):
                class_data = np.array(sorted_data[i]['data'])
                class_data = list(class_data * PNF[i])
                out_data += class_data
                class_label = [list(sorted_data[i]['label'])] * len(class_data)
                out_label += class_label
            out_data = np.array(out_data)
            out_label = np.array(out_label)
            indices = np.random.permutation(range(out_data.shape[0]))
            out_data = out_data[indices]
            out_label = out_label[indices]
        return out_data, out_label


def category(data, label):
    class_num = len(list(label[0]))
    sorted_data = {x: {'data': [], 'label': []} for x in range(class_num)}
    for ind, d in enumerate(data):
        sorted_data[np.argmax(label[ind])]['data'].append(d)
        sorted_data[np.argmax(label[ind])]['label'] = label[ind]
    return sorted_data


def abcd(sorted_data):
    """
    get a, b, c, d of all the features in sorted_data, the sorted_data will be a dict with key being the indices of
    class and value being a dict including 'data' and one hot 'label' of the class.
    a_{i,j} , b_{i,j} , c_{i,j} and d_{i,j} denote the document frequencies associated with the corresponding conditions
    a_{i,j}: document frequency of documents that belong to class Ci and contain term tj
    b_{i,j}: document frequency of documents that belong to class Ci but dont contain term tj
    c_{i,j}: document frequency of documents that dont belong to class Ci but contain term tj
    d_{i,j}: document frequency of documents that dont belong to class Ci and dont contain term tj
    :param sorted_data:a dict with key being the indices of class and value being a dict including 'data' and
    one hot 'label' of the class. the data is the features vectors where the values of certain dimensionality is the one
    hot value denotes the existence of this feature the frequency value of this features or the tfidf value of this
    feature. Note that the values can not be less than 0.
    :return: series of arrays with shape (num_class, num_features, 4), num_calss is the number of class in sorted_data,
    num_features denotes the number of features, which is determined by the sorted data; 4 denotes four values of a, b,
    c, d.
    """
    out = []
    for c in range(len(sorted_data)):
        vectors = np.array(sorted_data[c]['data'])
        vectors[vectors > 0] = 1  # transfer into one hot
        len_v = vectors.shape[0]
        nic_vectors = not_in_class_vectors(sorted_data, c)
        nic_vectors[nic_vectors > 0] = 1  # transfer into one hot
        len_n = nic_vectors.shape[0]
        a = np.sum(vectors, axis=0)
        b = len_v - a
        c = np.sum(nic_vectors, axis=0)
        d = len_n - c
        out_ = np.stack([a, b, c, d], axis=1)
        out.append(out_)
    return np.stack(out, axis=0)


def not_in_class_vectors(sorted_data, c):
    vectors = []
    for x in sorted_data.keys():
        if x != c:
            vectors += sorted_data[x]['data']
    return np.array(vectors)


if __name__ == '__main__':
    a = [[1, 2, 3], [1, 3, 4], [2, 4, 5], [2, 3, 4], [1, 2, 4], [1, 3, 5], [2, 4, 6]]
    b = [[0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0]]
    pnf = PNF()
    pnf.fit(a, b)
    data, label = pnf.pnf(a, b, ptype='1', shuffle=True)

