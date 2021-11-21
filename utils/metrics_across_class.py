"""
Tools to calculate mean value of F1-measure with respect to different size of classes.

author: Dongyang Yan
github: github.com/fiyen
lastest modification date: 2021/11/17
"""
import numpy as np
from utils import data_utils
from collections import Counter


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

# define the class groups with respect to the number of samples
class_number_range = [(0,10), (10, 100), (100, 500), (500, 99999)]
class_groups = {r: [] for r in class_number_range}
class_groups_f1 = {r: [] for r in class_number_range}

f1_file = open("f1.txt", 'r')
f1_scores = [line.split(',') for line in f1_file.readlines()]
for line in f1_scores:
  label, f1 = line[0], float(line[1])
  for low, up in class_number_range:
    if low <= sta_label_set[all_label[label]-1] < up:
      class_groups[(low, up)].append(label)
      class_groups_f1[(low, up)].append(float(f1))

for k, v in class_groups_f1.items():
  class_groups_f1[k] = sum(v) / len(v)
