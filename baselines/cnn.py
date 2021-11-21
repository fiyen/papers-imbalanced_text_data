"""
Pure cnn, with or without random oversampling.
author: Dongyang Yan
github: github.com/fiyen
last modification date: 2021/11/08
"""
from tensorflow import keras
from tensorflow.keras import layers
from utils import data_utils
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import metrics
from collections import Counter

'''physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)'''


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


if __name__ == "__main__":
    from utils.data_utils import transfer_imbalanced_dataset

    # Load data
    print("Loading data...")
    # if for_text_gen = True, training for text gen methods, else, training for cnn.ro or cnn.
    x_, y_, vocabulary, vocabulary_inv, test_size, all_label = data_utils.load_data(for_text_gen=False)
    all_label_inv = {}
    for key, val in all_label.items():
        all_label_inv[val] = key

    x_train, x_test = x_[:-test_size], x_[-test_size:]
    y_train, y_test = y_[:-test_size], y_[-test_size:]

    sta_label_set = class_num(y_train)
    sta_label_set = [i for i, _ in Counter(sta_label_set).most_common()]

    # For the datasets that is imbalanced, create the imbalanced condition.
    #x_train, y_train = transfer_imbalanced_dataset(x_train, y_train, [0,16,17,18,19], [48,54,56,46,37], seed=111)

    x_train, y_train = multiple_xy(x_train, y_train, 400)
    # x_test, y_test = multiple_xy(x_test, y_test, 500)
    # shuffle
    index_train = [i for i in range(x_train.shape[0])]
    random.shuffle(index_train)
    x_train = np.array([x_train[i] for i in index_train])
    y_train = np.array([y_train[i] for i in index_train])

    sequence_length = x_train.shape[-1]
    embedding_dimension = 300
    num_features = len(vocabulary_inv) + 100
    class_num = y_train.shape[1]

    print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))
    polar_num = 10
    model = keras.Sequential([
        layers.Embedding(input_dim=num_features, output_dim=embedding_dimension, input_length=sequence_length),
        #Embedding(words_list=vocabulary_inv, input_dim=num_features, output_dim=embedding_dimension,
        #           input_length=sequence_length, trainable=False, name='embedding_layer', fast_mode=True,
        #           filename='glove.6B.50d.txt', path='D:/onedrive/work/word_vector',
        #           preprocessed_vector_path='C:/test_output', mask_zero=True),
        layers.Conv1D(filters=50, kernel_size=5, strides=1, padding='valid', use_bias=True, activation='relu'),
        # layers.BatchNormalization(),
        #layers.Activation(activation='relu'),
        layers.MaxPool1D(2, padding='valid'),
        #layers.Conv1D(filters=100, kernel_size=3, strides=2, padding='valid', use_bias=True, activation=None),
        #layers.BatchNormalization(),
        #layers.Activation(activation='relu'),
        #layers.MaxPool1D(2, padding='valid'),
        #layers.Conv1D(filters=40, kernel_size=2, strides=2, padding='valid', use_bias=True, activation=None),
        #layers.BatchNormalization(),
        #layers.Activation(activation='relu'),
        #layers.MaxPool1D(2, padding='valid'),
        layers.Flatten(),
        layers.Dense(200, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(class_num, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    accuracy = []
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.1)
    evaluate = model.evaluate(x_test, y_test)
    print('evaluate, accuracy: ', evaluate)
    y_pred = np.array(model.predict_classes(x_test))
    y_true = np.array(np.argmax(y_test, axis=-1))
    f1_score = metrics.f1_score(y_true, y_pred, average='macro')
    print('f1 score macro:  ', f1_score)
    f1_score = metrics.f1_score(y_true, y_pred, average='micro')
    print('f1 score micro:  ', f1_score)
    f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
    print('f1 score weighted:  ', f1_score)
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    print('recall score macro:', recall)
    recall = metrics.recall_score(y_true, y_pred, average='micro')
    print('recall score micro:', recall)
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    print('precision score macro:', precision)
    precision = metrics.precision_score(y_true, y_pred, average='micro')
    print('precision score micro:', precision)

    f1_score = metrics.f1_score(y_true, y_pred, average=None)

    import xlwt
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Results', cell_overwrite_ok=True)
    ws.write(0, 1, 'CNN')
    row = 1
    for i in sta_label_set:
        print('label: {}, f1 measure: {}'.format(all_label_inv[i + 1], f1_score[i]))
        ws.write(row, 0, all_label_inv[i + 1])
        ws.write(row, 1, f1_score[i])
        row += 1
    wb.save('Results.xls')

    data = history.history
    with open('data.txt', 'a') as file_obj:
        file_obj.write(str(polar_num))
        file_obj.write(str(' '))
        for i in data['accuracy']:
            file_obj.write(str(i))
            file_obj.write(' ')
        file_obj.write('\n')
    accuracy.append(data['accuracy'])
    for acc in accuracy:
        plt.plot(acc)
    plt.show()
