"""
Tools to process the text data.
"""

from collections import Counter
import itertools
import chardet
import numpy as np
import re
import os
import random
from config import Config
args = Config()


def transfer_imbalanced_dataset(x, y, class_ids, sample_nums, seed=0):
  """

  :param x: data samples
  :param y: data labels, corresponding to x one by one.
  :param class_ids: the class that need to be transferred to minority class. Use the label to be the index of the class.
  :param sample_nums: The number of samples after transferred. Should with the same length as class_ids.
  :param seed: random seed.
  :return:
  """
  y_is_one_hot = True
  if isinstance(y[0], int) or isinstance(y[0], str):
    y_is_one_hot = False

  if y_is_one_hot:
    # if num_classes is none, infer it from y (assume that y is one-hot ndarray form.)
    sum_classes = np.sum(y, axis=0)
    num_classes = sum_classes.shape[0]
    sort_data = {i: {'data': [], 'labels': []} for i in range(num_classes)}
  else:
    # Generate labels
    all_label = dict()
    for label in y:
      if label not in all_label:
        all_label[label] = len(all_label)
    num_classes = len(all_label)
    sort_data = {i: {'data': [], 'labels': []} for i in range(num_classes)}
  for index, label in enumerate(y):
    if y_is_one_hot:
      i = np.argmax(label)
    else:
      i = all_label[label]
    sort_data[i]['data'].append(x[index])
    sort_data[i]['labels'].append(label)
  for index, num in zip(class_ids, sample_nums):
    data = sort_data[index]['data']
    labels = sort_data[index]['labels']
    random.seed(seed)
    random.shuffle(data)
    sort_data[index]['data'] = data[:num]
    sort_data[index]['labels'] = labels[:num]
  data = []
  labels = []
  for i in sort_data.keys():
    data += sort_data[i]['data']
    labels += sort_data[i]['labels']
  indexes = list(range(len(data)))
  random.shuffle(indexes)
  data = [data[i] for i in indexes]
  labels = [labels[i] for i in indexes]
  return np.array(data), np.array(labels)


def clean_str(string):
  """
  将文本中的特定字符串做修改和替换处理
  :param string:
  :return:
  """
  string = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", string)
  string = re.sub(r":", " : ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " ? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()


def load_data_and_labels(x_to_list=True, y_one_hot=True, for_text_gen=False):
  """
  Loads data from files, splits the data into words and generates labels.
  Returns split sentences and labels.
  :param x_to_list: whether make each text in x split into list of words.
  :param y_one_hot: whether return the y with one hot form.
  :param for_text_gen: whether use the generated dataset as the training data
  """
  # Load data from files
  folder_prefix = args.datasets_dir
  file_name = args.file_name
  file_form = args.file_form
  file_suffix = args.file_suffix
  full_path = os.path.join(folder_prefix, file_name)
  if for_text_gen:
    x_train = list(open(full_path + '-train-gen-' + file_form + file_suffix, 'r', encoding='utf8').readlines())
  else:
    x_train = list(open(full_path + '-train-' + file_form + file_suffix, 'r', encoding='utf8').readlines())
  x_test = list(open(full_path + '-test-' + file_form + file_suffix, 'r', encoding='utf8').readlines())
  test_size = len(x_test)
  x_text = x_train + x_test

  '''le = len(x_text)
  for i in range(le):
    encode_type = chardet.detect(x_text[i])
    x_text[i] = x_text[i].decode(encode_type['encoding'])  # 进行相应解码，赋给原标识符（变量'''
  y = [s.split()[0].split()[0] for s in x_text]
  x_text = [s.split()[1:] for s in x_text]
  if not x_to_list:
    x_text = [' '.join(x) for x in x_text]
  #x_text = [clean_str(sent) for sent in x_text]
  #x_text = [s.split()[1:] for s in x_text]

  '''x_text = [clean_str(sent) for sent in x_text]
  y = [s.split(' ')[0].split(':')[0] for s in x_text]
  x_text = [s.split(" ")[1:] for s in x_text]'''
  # Generate labels
  all_label = dict()
  for label in y:
    if label not in all_label:
      all_label[label] = len(all_label) + 1
  if y_one_hot:
    one_hot = np.identity(len(all_label))
    y = [one_hot[all_label[label]-1] for label in y]
  return x_text, y, test_size, all_label


def pad_sentences(sentences, padding_word="<PAD/>"):
  """
  Pads all sentences to the same length. The length is defined by the longest sentence.
  Returns padded sentences.
  """
  sequence_length = max(len(x) for x in sentences)
  padded_sentences = []
  for i in range(len(sentences)):
    sentence = sentences[i]
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    padded_sentences.append(new_sentence)
  return padded_sentences


def build_vocab(sentences):
  """
  Builds a vocabulary mapping from word to index based on the sentences.
  Returns vocabulary mapping and inverse vocabulary mapping.
  """
  # Build vocabulary
  word_counts = Counter(itertools.chain(*sentences))
  # Mapping from index to word
  # vocabulary_inv=['<PAD/>', 'the', ....]
  vocabulary_inv = [x[0] for x in word_counts.most_common()]
  # Mapping from word to index
  # vocabulary = {'<PAD/>': 0, 'the': 1, ',': 2, 'a': 3, 'and': 4, ..}
  vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
  return vocabulary, vocabulary_inv


def build_input_data(sentences, labels, vocabulary):
  """
  Maps sentences and labels to vectors based on a vocabulary.
  """
  x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
  y = np.array(labels)
  return x, y


def load_data(for_text_gen=False):
  """
  Loads and preprocessed data
  Returns input vectors, labels, vocabulary, and inverse vocabulary.
  """
  # Load and preprocess data
  sentences, labels, test_size, all_label = load_data_and_labels(for_text_gen=for_text_gen)
  sentences_padded = pad_sentences(sentences)
  vocabulary, vocabulary_inv = build_vocab(sentences_padded)
  x, y = build_input_data(sentences_padded, labels, vocabulary)
  return x, y, vocabulary, vocabulary_inv, test_size, all_label
