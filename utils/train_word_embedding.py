"""
To get the word embedding from large dataset.
"""
import chardet
from models.word_embedding import WordEmbedding as WE

folder_prefix = 'D:/data/imbalancedData/datasets/'
x_train = list(open(folder_prefix+"20ng-train-all-terms.txt", 'r', encoding='utf8').readlines())
x_test = list(open(folder_prefix+"20ng-test-all-terms.txt", 'r', encoding='utf8').readlines())
x_all = []
x_all = x_all + x_train + x_test
x_train = list(open(folder_prefix+"r52-train-all-terms.txt", 'r', encoding='utf8').readlines())
x_test = list(open(folder_prefix+"r52-test-all-terms.txt", 'r', encoding='utf8').readlines())
x_all = x_all + x_train + x_test
x_train = list(open(folder_prefix+"amazon-reviews-train-no-stop.txt", 'r', encoding='utf8').readlines())
x_test = list(open(folder_prefix+"amazon-reviews-test-no-stop.txt", 'r', encoding='utf8').readlines())
x_all = x_all + x_train + x_test
x_train = list(open(folder_prefix+"webkb-train-stemmed.txt", 'r', encoding='utf8').readlines())
x_test = list(open(folder_prefix+"webkb-test-stemmed.txt", 'r', encoding='utf8').readlines())
x_all = x_all + x_train + x_test
x_train = list(open(folder_prefix+"emotions-train-all-terms.txt", 'r', encoding='utf8').readlines())
x_test = list(open(folder_prefix+"emotions-test-all-terms.txt", 'r', encoding='utf8').readlines())
x_all = x_all + x_train + x_test
'''le = len(x_all)
for i in range(le):
    encode_type = chardet.detect(x_all[i])
    x_all[i] = x_all[i].decode(encode_type['encoding'])  # 进行相应解码，赋给原标识符（变量
x_all = [s.split()[1:] for s in x_all]'''

we = WE(size=300, epochs=100, min_count=0, save=True, fname='all_w2v')
we.fit_train(x_all)

# 将训练得到的向量存储为txt格式
vocab = we.w2v.wv.index_to_key
file = open('D:/data/imbalancedData/w2v/vectors.txt', 'w', encoding='utf8')
for v in vocab:
    file.write(v + ' ')
    for i in we.w2v.wv[v]:
        file.write(str(i) + ' ')
    file.write('\n')
file.close()
