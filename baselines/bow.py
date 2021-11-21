#---------------------import---------------------
import time
from sklearn import svm
import numpy as np
from sklearn import metrics
from collections import Counter

# ------------------------function------------------------
# accuracy
def accuracy(predict_label, valid_label):
    is_correct = np.array(predict_label) == np.array(valid_label)
    return sum(is_correct) / len(is_correct)


if __name__ == '__main__':
    from baselines import pattern
    from config import Config
    args = Config()
    from utils.data_utils import transfer_imbalanced_dataset

    source_path = args.datasets_dir + '/'
    file_name_valid = args.file_name + '-test-' + args.file_form + args.file_suffix
    file_name_train = args.file_name + '-train-' + args.file_form + args.file_suffix

    train_data = []
    train_labels = []
    valid_data = []
    valid_labels = []
    for line in open(source_path + file_name_valid, 'r', encoding='utf8'):
        content = line.split()
        valid_labels.append(content[0])
        valid_data.append(line.replace(content[0], ''))

    for line in open(source_path + file_name_train, 'r', encoding='utf8'):
        content = line.split()
        train_labels.append(content[0])
        train_data.append(line.replace(content[0], ''))

    # For the datasets that is imbalanced, create the imbalanced condition.
    train_data, train_labels = transfer_imbalanced_dataset(train_data, train_labels, [0,16,17,18,19], [48,54,56,46,37], seed=0)
    # train_data, train_labels = transfer_imbalanced_dataset(valid_data, valid_labels, [2], [33], seed=999)
    train_data = train_data.tolist()
    train_labels = train_labels.tolist()

    # 读取和处理20news group
    '''source_path = 'D:/20news'
    categories = os.listdir(source_path)
    data = []
    for c in categories:
        f_list = os.listdir(source_path+'/'+c)
        for f in f_list:
            for line in open(source_path+'/'+c+'/'+f, 'r', encoding='gbk'):
                data.append([c, line])'''


    from baselines import smote as se
    from baselines.pnf import PNF
    from baselines.bdsk import BDSK
    from baselines.rwo import RWO

    # 特征表示
    start = time.time()
    ##########################################Parameters##################################
    pattern_name = 'tf'
    ######################################################################################
    # 数据增广
    ####################################Parameters###################################
    expand = "PNF"
    cut_num = args.cut_off_num
    #################################################################################

    print('pattern is ', pattern_name)
    if pattern_name == 'tf':
        count_data = pattern.to_tfidf_vector(train_data, valid_data, norm='l2', num_features=3000)
    elif pattern_name == 'tfidf':
        count_data = pattern.to_tfidf_vector(train_data, valid_data, norm='l2', num_features=3000)
    else:
        raise KeyError('pattern error')
    print('cost time: {:.2f}s.'.format(time.time() - start))
    x_train = count_data[0].toarray()
    x_valid = count_data[1].toarray()
    len_t = len(train_labels)
    labels, label_set = pattern.to_categorical_label(train_labels + valid_labels)
    label_set_inv = {}
    for ind, val in enumerate(label_set):
        label_set_inv[ind] = val
    y_train = np.array(labels[:len_t])
    y_valid = np.array(labels[len_t:])
    sta_label_set = pattern.class_num(y_train)
    sta_label_set = [i for i, _ in Counter(sta_label_set).most_common()]

    if expand == 'RO':
        print('data expand: ', expand)
        x_train, y_train = se.multiple_xy(x_train, y_train, cut_num=cut_num)
    elif expand == 'SMOTE':
        print('data expand: ', expand)
        smote = se.SMOTE(5)
        x_train, y_train = smote.data_expand(x_train, y_train, cut_num=cut_num)
    elif expand == 'PNF':
        print('data expand: ', expand)
        pnf = PNF()
        pnf.fit(x_train, y_train)
        x_train, y_train = pnf.pnf(x_train, y_train, ptype='1', shuffle=True)
        x_valid, y_valid = pnf.pnf(x_valid, y_valid, ptype='1', shuffle=True)
    elif expand == 'BDSK':
        print('data expand: ', expand)
        bdsk = BDSK()
        x_train, y_train = bdsk.expand_data(bdsk.fit(x_train, y_train), cut_num=cut_num, k_neighbor=None)
    elif expand == 'RWO':
        print('data expand: ', expand)
        rwo = RWO()
        x_train, y_train = rwo.data_expand(x_train, y_train, cut_num=cut_num)
    else:
        print('data expand: No operation complemented.')

    import xlwt
    column = 1
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Results', cell_overwrite_ok=True)
    c_bags = [0.1, 0.5, 1, 2, 5, 10]
    for c in c_bags:
        print('c: ', c)
        svc_c = svm.SVC(C=c)
        svc_c.fit(x_train, np.argmax(y_train, -1))
        pred = svc_c.predict(x_valid)
        print('count data, svm, accuracy: ', accuracy(pred, np.argmax(y_valid, -1)))
        f1_score = metrics.f1_score(np.argmax(y_valid, -1), pred, average='macro')
        print('BOW, SVM, f1 score macro: ', f1_score)
        f1_score = metrics.f1_score(np.argmax(y_valid, -1), pred, average='micro')
        print('BOW, SVM, f1 score micro: ', f1_score)
        f1_score = metrics.f1_score(np.argmax(y_valid, -1), pred, average='weighted')
        print('BOW, SVM, score weighted: ', f1_score)
        f1_score = metrics.f1_score(np.argmax(y_valid, -1), pred, average=None)
        recall = metrics.recall_score(np.argmax(y_valid, -1), pred, average='macro')
        print('BOW, SVM, recall macro: ', recall)
        recall = metrics.recall_score(np.argmax(y_valid, -1), pred, average='micro')
        print('BOW, SVM, recall micro: ', recall)
        precision = metrics.precision_score(np.argmax(y_valid, -1), pred, average='macro')
        print('BOW, SVM, precision macro: ', precision)
        precision = metrics.precision_score(np.argmax(y_valid, -1), pred, average='micro')
        print('BOW, SVM, precision micro: ', precision)

        row = 0
        ws.write(row, column, pattern_name + '-' + expand)
        row += 1
        for i in sta_label_set:
            print('label: {}, f1 measure: {}'.format(label_set_inv[i], f1_score[i]))
            if column == 1:
                ws.write(row, 0, label_set_inv[i])
            ws.write(row, column, f1_score[i])
            row += 1
        wb.save('Results.xls')
