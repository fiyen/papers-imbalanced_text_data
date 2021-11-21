"""
to transfer a text into vector, including the feature extraction and selection.
: to_count_vector: digit value of each dimension is the frequency of words
: to_tfidf_vector: ...... is the tfidf value of words
: to_net_vector: ...... is the value of network attribute for each node (corresponding to specific word)

author: Dongyang Yan
github: github.com/fiyen
last modification date: 2021/11/12
"""


# ---------------------------import--------------------------
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import networkx as nx
import numpy as np
import math
from scipy import sparse
from sklearn.preprocessing import normalize
from multiprocessing import pool, cpu_count
from collections import Counter
import itertools


# ---------------------------function------------------------
# 转换成计数向量
def to_count_vector(train_data, valid_data, norm='', ngram_range=(1, 1), num_features=None):
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=ngram_range, max_features=num_features)
    count_vect.fit(train_data)
    vector_train = count_vect.transform(train_data)
    vector_valid = count_vect.transform(valid_data)
    if norm is 'l1':
        return normalize(vector_train, norm='l1'), normalize(vector_valid, norm='l1')
    elif norm is 'l2':
        return normalize(vector_train, norm='l2'), normalize(vector_valid, norm='l2')
    else:
        return vector_train, vector_valid


# 转换成tfidf向量
def to_tfidf_vector(train_data, valid_data, norm='l2', num_features=None):
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', norm=norm, max_features=num_features)
    tfidf_vect.fit(train_data)
    vector_train = tfidf_vect.transform(train_data)
    vector_valid = tfidf_vect.transform(valid_data)
    return vector_train, vector_valid


# 转换成网络特征向量
def preprocess(train_data, valid_data):
    '''
    预处理，将数据转换成文字编号, 将编号按照词语出现的频率排序
    :param train_data:支持raw data (原始文档，每行是string格式)或处理后的list
    :param valid_data:同上
    :return:
    '''
    try:
        train_data = [line.split() for line in train_data]
        valid_data = [line.split() for line in valid_data]
    except AttributeError:
        pass
    word_counts = Counter(itertools.chain(*train_data))
    vocabulary = [x[0] for x in word_counts.most_common()]
    words_code = {x: i for i, x in enumerate(vocabulary)}
    train_data_coded = []
    for line in train_data:
        data_coded_row = []
        for element in line:
            data_coded_row.append(words_code[element])
        train_data_coded.append(data_coded_row)
    valid_union_words = [] # 这行开始求valid_data的编码，与train相比多出的元素用负值标注
    for line in valid_data:
        valid_union_words = set(valid_union_words).union(line)
    extra_words = set(valid_union_words).difference(vocabulary)
    extra_words_code = {}
    for index, element in enumerate(extra_words):
        extra_words_code[element] = -(index + 1)
    valid_data_coded = []
    for line in valid_data:
        valid_data_row = []
        for element in line:
            if element in extra_words:
                valid_data_row.append(extra_words_code[element])
            else:
                valid_data_row.append(words_code[element])
        valid_data_coded.append(valid_data_row)
    return train_data_coded, valid_data_coded, vocabulary


def net_construct(line_coded): # total 为训练集的所有单词数
    G = nx.Graph()
    G.clear()
    for index, element in enumerate(line_coded):
        if index == 0:
            G.add_node(element)
        else:
            G.add_node(element)
            G.add_edge(element, line_coded[index - 1])
    return G


def to_net_vector(train_data, valid_data, feature='degree', norm='', fast_mode=True, length=0, threads=0): # 用词共现网络，无向图
    feature_bags = ['degree', 'clustering', 'eccentricity', 'eigenvector',
                    'page_rank', 'accessibility', 'concentric_symmetry']
    if feature not in feature_bags:
        raise Exception('error!, no feature named:', feature)
    train_data_coded, valid_data_coded, vocabulary = preprocess(train_data, valid_data)
    if length > 0:
        length_vector = length
    else:
        length_vector = len(vocabulary)
    #row_data = preprocessing(train_data, valid_data)
    #train_data_coded = row_data[0]
    #valid_data_coded = row_data[1]
    #length_vector = row_data[2]
    if not fast_mode:
        vector_train = []
        count = 0
        for line in train_data_coded:
            net = net_construct(line)
            vector_line = blender(net, feature)
            # vector_line = normalization(blender(net, length_vector, feature))
            # vector_train.append(vector_line)
            vector_train = to_csr_matrix(vector_train, vector_line, length_vector)
            # count += 1
            # if count % 100 == 0:
            #     print(count)
        vector_valid = []
        for line in valid_data_coded:
            net = net_construct(line)
            vector_line = blender(net, feature)
            # vector_line = normalization(blender(net, length_vector, feature))
            vector_valid = to_csr_matrix(vector_valid, vector_line, length_vector)
            # count += 1
            # if count % 100 == 0:
            #     print(count)'''
    else:
        if threads <= 0:
            threads = cpu_count()
        p = pool.Pool(threads)
        zip_train_input = zip(train_data_coded, [feature] * len(train_data_coded))
        zip_valid_input = zip(valid_data_coded, [feature] * len(valid_data_coded))
        vector_train = p.map(_swap_for_par, zip_train_input)
        vector_valid = p.map(_swap_for_par, zip_valid_input)
        p.close()
        p.join()
        if len(train_data) > 0:
            vector_train = to_csr_matrix_v2(vector_train, length_vector)
        else:
            vector_train = []
        if len(valid_data) > 0:
            vector_valid = to_csr_matrix_v2(vector_valid, length_vector)
        else:
            vector_valid = []
    print('to_net_vector done!')
    if norm is 'l1':
        return normalize(vector_train, norm='l1'), normalize(vector_valid, norm='l1')
    elif norm is 'l2':
        return normalize(vector_train, norm='l2'), normalize(vector_valid, norm='l2')
    else:
        return vector_train, vector_valid


def _par_for_net(line, feature):
    """
    辅助to_net_vector函数，实现并行操作
    :param line:
    :param length_vector:
    :param feature:
    :return:
    """
    net = net_construct(line)
    vector_line = blender(net, feature)
    return vector_line


def _swap_for_par(args):
    return _par_for_net(*args)


def blender(G, feature):# 结合G和feature，生成所需向量,长度为length_vector(改：直接返回字典）
    feature_bags = ['degree', 'clustering', 'eccentricity', 'eigenvector',
                    'page_rank', 'accessibility']
    if feature not in feature_bags:
        raise Exception('error!, no feature named:', feature)

    #vector_G = [0 for col in range(length_vector)]
    if feature == 'degree':
        value_feature = {}
        for key, element in degree(G).items():
            value_feature[key] = element
        '''for node in nx.nodes(G):
            if node >= 0:
                vector_G[node] = value_feature[node]'''
    elif feature == 'clustering':
        value_feature = clustering(G)
        # print(value_feature)
        '''for node in nx.nodes(G):
            if node >= 0:
                vector_G[node] = value_feature[node]'''
    elif feature == 'eccentricity':
        value_feature = eccentricity(G)
        '''for node in nx.nodes(G):
            if node >= 0:
                vector_G[node] = value_feature[node]'''
    elif feature == 'eigenvector':
        value_feature = eigenvector(G)
        '''for node in nx.nodes(G):
            if node >= 0:
                vector_G[node] = value_feature[node]'''
    elif feature == 'page_rank':
        value_feature = page_rank(G)
        '''for node in nx.nodes(G):
            if node >= 0:
                vector_G[node] = value_feature[node]'''
    elif feature == 'accessibility':
        value_feature = accessibility(G)
        '''for node in nx.nodes(G):
            if node >= 0:
                vector_G[node] = value_feature[node]'''
    return value_feature


#b标准化
def normalization(data):
    '''data为字典'''
    sum_data = sum(data.values())
    if sum_data == 0:
        return data
    else:
        for key, value in data.items():
            data[key] = value / sum_data
            return data


def degree(G):
    degree_G = {}
    for key, ele in G.degree:
        degree_G[key] = ele
    if degree_G:
        return degree_G
    else:
        return {0:0}


def clustering(G):
    c_t = nx.clustering(G)
    if c_t:
        return c_t
    else:
        return {0:0}


def eccentricity(G):
    e = {}
    # e_t = nx.eccentricity(G)
    e_t = nx.nodes(G)
    if e_t:
        # return nx.eccentricity(G)
        for n in G.nbunch_iter():
            length = nx.single_source_shortest_path_length(G, n)
            e[n] = max(length.values())
        return e
    else:
        return {0:0}


def eigenvector(G):
    if len(G.nodes) != 0:
        return nx.eigenvector_centrality(G, max_iter=10000)
    else:
        return {0:0}


def page_rank(G):
    p_r = nx.pagerank(G)
    if p_r:
        return p_r
    else:
        return {0:0}


def accessibility(G, k=2): # k=1 or 2 or 3
    accessibility_ = {}
    if len(G.nodes) == 0:
        return {0:0}
    if len(G.nodes) == 1:
        for node in G.nodes:
            accessibility_[node] = 1
        return accessibility_
    for node in G.nodes:
        endnodes = get_path_endnode(G, node, k)
        endnodes_bags = set(endnodes).union()
        accessibility_sole = 0
        for element in endnodes_bags: # 计算accessibility值
            number_of_element = endnodes.count(element)
            possibility_of_element = number_of_element / len(endnodes)
            accessibility_sole += -possibility_of_element*math.log(possibility_of_element, math.e)
        accessibility_[node] = math.exp(accessibility_sole)
    return accessibility_


def get_neighbors(G, n):
    neighbors = []
    for node in G.neighbors(n):
        neighbors.append(node)
    return neighbors


def get_path_endnode(G, n, k): #得到k步随机漫步的终点，无环路
    path = []
    if k == 1:
        for node_1 in G.neighbors(n):
            path.append(node_1)
        return path
    if k == 2:
        for node_1 in G.neighbors(n):
            for node_2 in G.neighbors(node_1):
                if node_2 != n:
                    path.append(node_2)
        return path
    if k == 3:
        for node_1 in G.neighbors(n):
            for node_2 in G.neighbors(node_1):
                for node_3 in G.neighbors(node_2):
                    if len(set([n]).union([node_1, node_2, node_3])) == 4:
                        path.append(node_3)
        return path


# 将矩阵转化成稀疏矩阵csr_matrix, 一行一行增加数据的情况，data_row为一个字典
def to_csr_matrix(old_csr_matrix, data_row, length_row):
    if old_csr_matrix == []:
        data = []
        indices = []
        indptr = [0]
        for key in data_row:
            if 0 <= key < length_row and data_row[key] != 0:
                data.append(data_row[key])
                indices.append(key)
        indptr.append(len(data))
        return sparse.csr_matrix((data, indices, indptr), shape=(1, length_row))
    else:
        indptr = old_csr_matrix.indptr.tolist()
        indices = old_csr_matrix.indices.tolist()
        data = old_csr_matrix.data.tolist()
        number_row = old_csr_matrix._shape[0] + 1
        number_col = old_csr_matrix._shape[1]
        for key in data_row:
            if 0 <= key < length_row and data_row[key] != 0:
                data.append(data_row[key])
                indices.append(key)
        indptr.append(len(data))
        return sparse.csr_matrix((data, indices, indptr), shape=(number_row, number_col))


def to_csr_matrix_v2(dict_data, length_vector):
    """
    将list_data转换成稀疏矩阵
    :param length_vector:
    :param dict_data: list, 元素为dict
    :return:
    """
    shape = (len(dict_data), length_vector)
    data = [line[key] for line in dict_data for key in line.keys() if line[key] != 0 and 0 <= key < length_vector]
    data = np.array(data)
    row_col = [[row, col] for row, dict_ in enumerate(dict_data) for col in dict_.keys() if dict_data[row][col] != 0
               and 0 <= col < length_vector]
    row_col = np.array(row_col)
    return sparse.csr_matrix((data, (row_col[:, 0], row_col[:, 1])), shape=shape)

# pattern混合模式,函数中用pattern_1 = 'pattern_1', pattern_2 = 'pattern_2'的形式输入pattern
def mix_pattern(G, proportion, pattern):
    '''几种pattern所求值的线性组合，proportion中为各pattern的比例，写成一个字典'''
    feature_bags = ['degree', 'clustering', 'eccentricity', 'eigenvector',
                    'page_rank', 'accessibility']
    for ele in pattern.values():
        if ele not in feature_bags:
            raise Exception('there is no feature: {}'.format(ele))

    func_dict = {'degree': degree, 'clustering':clustering, 'eccentricity':eccentricity,
                 'eigenvector':eigenvector, 'page_rank':page_rank, 'accessibility':accessibility}
    results = {}
    for ele in pattern.values():
        results[ele] = normalization(func_dict.get(ele)(G))
    keys_results = results.keys()
    for key in keys_results:
        keys_G = results[key].keys()
        break
    results_mix = {}
    # print(results)
    for key in keys_G:
        values_of_key_G_mix = 0
        for key_ in keys_results:
            # print('key_ = {0}; key = {1}'.format(key_, key))
            values_of_key_G_mix += results[key_][key] * proportion[key_]
        results_mix[key] = values_of_key_G_mix
    return results_mix


def to_net_vector_mix(train_data, valid_data, proportion, **feature):
    feature_bags = ['degree', 'clustering', 'eccentricity', 'eigenvector',
                    'page_rank', 'accessibility', 'concentric_symmetry']
    for ele in feature.values():
        if ele not in feature_bags:
            raise Exception('there is no feature: {}'.format(ele))

    train_data_coded, valid_data_coded, length_vector = preprocess(train_data, valid_data)
    # row_data = preprocessing(train_data, valid_data)
    # train_data_coded = row_data[0]
    # valid_data_coded = row_data[1]
    # length_vector = row_data[2]
    vector_train = []
    count = 0
    for line in train_data_coded:
        net = net_construct(line)
        vector_line = mix_pattern(net, proportion, feature)
        # vector_train.append(vector_line)
        vector_train = to_csr_matrix(vector_train, vector_line, length_vector)
        # count += 1
        # if count % 100 == 0:
        #     print(count)
    vector_valid = []
    for line in valid_data_coded:
        net = net_construct(line)
        vector_line = mix_pattern(net, proportion, feature)
        vector_valid = to_csr_matrix(vector_valid, vector_line, length_vector)
        # count += 1
        # if count % 100 == 0:
        #     print(count)
    print('to_net_vector_mix done!')
    return vector_train, vector_valid


# 新的混合方法，两种feature以上的混合，先把所有feature 的向量都求出来，再根据proportion计算
def all_vector(train_data, valid_data):
    '''计算并返回所有feature对应的向量'''
    vector_all = {}
    feature_bags = ['degree', 'clustering', 'eccentricity', 'eigenvector',
                    'page_rank', 'accessibility']
    for feature in feature_bags:
        vector_all[feature] = to_net_vector(train_data, valid_data, feature)
        print('feature ' + feature + ' done!')
    return vector_all


def new_to_net_vector_mix(vector_all, proportion): # 输出结果为array
    feature_bags = ['degree', 'clustering', 'eccentricity', 'eigenvector',
                    'page_rank', 'accessibility']

    vector_train = 0
    vector_valid = 0
    for feature in feature_bags:
        vector_train = proportion[feature] * vector_all[feature][0].toarray() + vector_train
        vector_valid = proportion[feature] * vector_all[feature][1].toarray() + vector_valid
    return vector_train, vector_valid


# 结合count计数的net方法
def to_net_count_vector(train_data, valid_data, feature='degree',norm='l1'):
    count_vector = to_count_vector(train_data, valid_data,norm=norm)
    net_vector = to_net_vector(train_data, valid_data, feature=feature, norm=norm)
    if norm is 'l1':
        return (count_vector[0] + net_vector[0])*0.5, (count_vector[1]+net_vector[1])*0.5
    elif norm is 'l2':
        return np.sqrt(count_vector[0].multiply(count_vector[0]) + net_vector[0].multiply(net_vector[0])),\
               np.sqrt(count_vector[1].multiply(count_vector[1]) + net_vector[1].multiply(net_vector[1]))


# 分子加权的tfidf的net方法
def to_net_tfidf_vector(train_data, valid_data, feature='degree',norm='l1'):
    tfidf_vector = to_tfidf_vector(train_data, valid_data, norm=norm)
    net_vector = to_net_vector(train_data, valid_data, feature=feature, norm=norm)
    if norm is 'l1':
        return (tfidf_vector[0] + net_vector[0])*0.5, (tfidf_vector[1]+net_vector[1])*0.5
    elif norm is 'l2':
        return np.sqrt(tfidf_vector[0].multiply(tfidf_vector[0]) + net_vector[0].multiply(net_vector[0])),\
               np.sqrt(tfidf_vector[1].multiply(tfidf_vector[1]) + net_vector[1].multiply(net_vector[1]))


# 分子分母加权的tfidf的net方法
def to_net_tfidf2_vector(train_data, valid_data, feature='degree'):
    return 0


def to_categorical_label(label):
    """
    把label转换成one hot编码
    """
    length = len(label)
    max_index = len(set(label))
    label_set = [l for l in set(label)]
    label_index = {l: i for i, l in enumerate(label_set)}
    code = np.zeros((length, max_index)).tolist()
    for i, l in enumerate(label):
        code[i][label_index[l]] = 1
    return code, label_set


def class_num(label):
    """
    统计出label里各标签的数量
    :param label: one_hot
    :return:
    """
    num = len(list(label[0]))
    label_set = {i: 0 for i in range(num)}
    for line in label:
        label_set[np.argmax(line)] += 1
    return label_set
