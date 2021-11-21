"""
oversampling by text generation methods.
implemented on googlelab, cause the computer processing is too slow (GTX2060).

author: Dongyang Yan
github: github.com/fiyen
last modification date: 2021/11/13
"""

from baselines.textgenrnn.textgenrnn import textgenrnn
import random
import os
from config import Config
import re
import json
args = Config()
import sentencepiece
spm_en = sentencepiece.SentencePieceProcessor("spm.model")


def train_new_textgen(spm):
  """
  train a new textgenrnn model on word level, with the help of sentencepiece to get the
  encoding of subwords.
  :param texts:
  :return:
  """
  # use the datasets in textgenrnn/datasets
  texts = []
  filelists = os.listdir("textgenrnn/datasets")
  filelists = [file for file in filelists if file.endswith('.txt') or file.endswith('csv')]
  for file in filelists:
    texts += open(os.path.join("textgenrnn/datasets", file), 'r', encoding='utf8').readlines()
  texts = text_encode_pieces_with_spm(texts, spm)
  textgen = textgenrnn()
  textgen.train_new_model(texts, num_epochs=100, gen_epochs=10, batch_size=1024)
  textgen.save("textgenrnn/weights/my_model.hdf5")
  json.dump(textgen.tokenizer.word_index, open('textgenrnn/weights/my_vocab.json', 'w'))


def text_encode_pieces_with_spm(texts, spm):
  """
  encode the text into pieces with sentencepiece model.
  :param texts:
  :return:
  """
  new_texts = []
  lengths = []
  for text in texts:
    text = spm.encode_as_pieces(text)
    lengths.append(len(text))
    text = ' '.join(text)
    new_texts.append(text)
  return new_texts, lengths


def text_decode_with_spm(texts):
  """
  decode the text pieces using sentencepiece model.
  :param texts:
  :param spm:
  :return:
  """
  new_texts = []
  for text in texts:
    text = text.strip().split()
    text = ''.join(text)
    text = re.sub('▁', ' ', text)
    new_texts.append(text.strip())
  return new_texts


def textgen_oversampling(class_data, gen_num, spm):
  """
  use textgenrnn to generate samples for class_name.
  :param class_data: samples in certain class.
  :param gen_num: number of new samples to be generated.
  :return:
  """
  textgen = textgenrnn(weights_path="textgenrnn/weights/my_model.hdf5",
                       vocab_path="textgenrnn/weights/my_vocab.json")
  class_data, lengths = text_encode_pieces_with_spm(class_data, spm)
  batch_size = min(64, sum(lengths)+1)
  textgen.train_on_texts(texts=class_data, batch_size=batch_size, num_epochs=20, gen_epochs=1000)
  new_texts = textgen.generate(gen_num, return_as_list=True)
  new_texts = text_decode_with_spm(new_texts)
  return new_texts


def balance_datasets(datasets, labels, spm, cut_num=None):
  """
  generate samples to make the datasets balanced (number of samples for each class is equal).
  :param datasets:
  :param labels:
  :param cut_num: number of samples for each class doesn't exceed cut_num if cut_num is not None.
  :return:
  """
  sort_data = {}
  for text, label in zip(datasets, labels):
    if label in sort_data.keys():
      sort_data[label].append(text)
    else:
      sort_data[label] = [text]
  if cut_num is None:
    cut_num = max([len(d) for d in sort_data.values()])
  for i, label in enumerate(sort_data.keys()):
    print("Begin text generation of label: %s -- %d/%d."%(label, i+1, len(sort_data)))
    texts = sort_data[label]
    gen_num = cut_num - len(texts)
    if gen_num > 0:
      new_texts = textgen_oversampling(texts, gen_num, spm)
      sort_data[label] += new_texts
    # save the new file
    save_path = "%s/%s-train-gen-%s%s" % (args.datasets_dir, args.file_name, args.file_form, args.file_suffix)
    with open(save_path, 'a', encoding='utf8') as f:
      for text in sort_data[label]:
        f.write("%s\t%s\n" % (label, text))
      f.close()
  data = []
  labels = []
  for label in labels:
    texts = sort_data[label]
    data += texts
    labels += [label] * len(texts)
  indexes = list(range(len(data)))
  random.shuffle(indexes)
  data = [data[i] for i in indexes]
  labels = [labels[i] for i in indexes]
  return data, labels


if __name__ == "__main__":
  #train_new_textgen(spm_en)
  from utils.data_utils import load_data_and_labels, transfer_imbalanced_dataset

  samples, labels, test_size, all_label = load_data_and_labels(x_to_list=False, y_one_hot=False)
  x_train, y_train = samples[:-test_size], labels[:-test_size]
  x_test, y_test = samples[-test_size:], labels[-test_size:]

  # For the datasets that is imbalanced, create the imbalanced condition.
  x_train, y_train = transfer_imbalanced_dataset(x_train, y_train, [0,16,17,18,19], [48,54,56,46,37], seed=111)

  x_train, y_train = balance_datasets(x_train, y_train, spm_en, cut_num=None)
