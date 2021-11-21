# Load data
import numpy as np
from collections import Counter
import random
from models.word_embedding import WordEmbedding
from models.random_walk import PathGen
from utils.data_utils import load_data
import tensorflow as tf
from tensorflow.keras import Sequential
from models.ncnn import PcnnNet
from config import Config
args = Config()


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


if __name__ == '__main__':
  import os
  from utils.data_utils import transfer_imbalanced_dataset
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


  print("Loading data...")
  x_, y_, vocabulary, vocabulary_inv, test_size, all_label = load_data()
  all_label_inv = {}
  for key, val in all_label.items():
    all_label_inv[val] = key
  #x_test, x_train = x_[:-test_size], x_[-test_size:]
  #y_test, y_train = y_[:-test_size], y_[-test_size:]
  x_train, x_test = x_[:-test_size], x_[-test_size:]
  y_train, y_test = y_[:-test_size], y_[-test_size:]
  # delete the sample with 0 element.
  y_train = np.array([y for i, y in enumerate(y_train) if sum(x_train[i]) > 0])
  x_train = np.array([x for x in x_train if sum(x) > 0])
  y_test = np.array([y for i, y in enumerate(y_test) if sum(x_test[i]) > 0])
  x_test = np.array([x for x in x_test if sum(x) > 0])
  # For the datasets that is imbalanced, create the imbalanced condition.
  x_train, y_train = transfer_imbalanced_dataset(x_train, y_train, [2], [33], seed=999)

  sta_label_set = class_num(y_train)
  sta_label_set = [i for i, _ in Counter(sta_label_set).most_common()]

  # Transform inputs and shuffle
  we = WordEmbedding(size=args.embedding_dimension, min_count=0, epochs=args.word2vec_training_epochs, save=args.save_option,
                     load=args.load_option, fname=args.fname)
  if args.save_option:
    we.fit_train([[str(x) for x in line] for line in x_train] + [[str(x) for x in line] for line in x_test])
  get_path = PathGen(args.polar_size, args.sequence_length, threads=args.threads4gen, expansion_ratio=args.expansion_ratio,
                     walk=args.walk, except_padding=args.padding, adaptive_expansion=args.adaptive_expansion,
                     cut_off_num=args.cut_off_num, sample_cross=args.sample_cross, undersampling=args.undersampling, p=args.p, q=args.q)
  x_train, y_train = get_path(x_train, y_train, verbose=True, batch_size=args.batch_size)
  # shuffle
  index_train = [i for i in range(x_train.shape[0])]
  random.shuffle(index_train)
  x_train = np.array([x_train[i] for i in index_train])
  y_train = np.array([y_train[i] for i in index_train])

  # Training and Validation
  # model, compile, fit

  #class_number =  2#@param {type:'integer'}
  class_number = y_train.shape[1]
  num_features = len(vocabulary_inv) + 100
  model = PcnnNet(output_dim=class_number, num_features=num_features, sequence_length=args.sequence_length, dict_source=args.dict_source,
            embedding_dim=args.embedding_dimension, words_list=vocabulary_inv, word_vectors=we.w2v, labeled=args.labeled,
            polar=(args.polar_size, 2, 2), conv_1=args.conv_1, polars=None,
            conv_2=args.conv_2, pooling=args.pooling, pooling_strides=args.pooling_strides)
  model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])
  model.build(input_shape=x_train.shape)
  #model.summary()
  history = model.fit(x_train, y_train, epochs=args.epochs, validation_split=args.validation_split)

  # Evaluate for normal prediction
  get_path = PathGen(args.polar_size, args.sequence_length, threads=args.threads4gen, walk=args.walk,
                     except_padding=args.padding, p=args.p, q=args.q)
  x_test_, y_test_ = get_path(x_test, y_test, verbose=True)

  evaluate = model.evaluate(x_test_, y_test_)
  print('evaluate, accuracy: ', evaluate)

  from sklearn import metrics
  import xlwt
  wb = xlwt.Workbook()
  ws = wb.add_sheet('Results', cell_overwrite_ok=True)

  y_pred = model.predict(x_test_)
  y_pred = np.argmax(y_pred, axis=-1)
  y_true = np.argmax(y_test_, axis=-1)
  f1_score = metrics.f1_score(y_true, y_pred, average='macro')
  print('f1 score macro: ', f1_score)
  f1_score = metrics.f1_score(y_true, y_pred, average='micro')
  print('f1 score micro: ', f1_score)
  f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
  print('f1 score weighted: ', f1_score)
  f1_score = metrics.f1_score(y_true, y_pred, average=None)
  recall = metrics.recall_score(y_true, y_pred, average='macro')
  print('recall score macro:', recall)
  recall = metrics.recall_score(y_true, y_pred, average='micro')
  print('recall score micro:', recall)
  precision = metrics.precision_score(y_true, y_pred, average='macro')
  print('precision score macro:', precision)
  precision = metrics.precision_score(y_true, y_pred, average='micro')
  print('precision score micro:', precision)

  ws.write(0, 1, 'NCNN')
  row = 1
  for i in sta_label_set:
      print('label: {}, f1 measure: {}'.format(all_label_inv[i+1], f1_score[i]))
      ws.write(row, 0, all_label_inv[i+1])
      ws.write(row, 1, f1_score[i])
      row += 1


  # Evaluate for selected prediction
  ratio = 5 #@param {type:'integer'}
  print('ave selected prediction:')
  acc, y_pred = model.evaluate_select(x_test, y_test, ratio=ratio, method='ave', walk=args.walk, batch=256,
                                      except_padding=args.padding, p=args.p, q=args.q)

  y_pred = np.array(y_pred)
  y_true = np.argmax(y_test, axis=-1)
  f1_score_ave = metrics.f1_score(y_true, y_pred, average='macro')
  print('f1 score macro for ave selecting', f1_score_ave)
  f1_score_ave = metrics.f1_score(y_true, y_pred, average='micro')
  print('f1 score micro for ave selecting', f1_score_ave)
  f1_score_ave = metrics.f1_score(y_true, y_pred, average='weighted')
  print('f1 score weighted for ave selecting', f1_score_ave)
  f1_score = metrics.f1_score(y_true, y_pred, average=None)
  recall_ave = metrics.recall_score(y_true, y_pred, average='macro')
  print('recall score macro for ave selecting', recall_ave)
  recall_ave = metrics.recall_score(y_true, y_pred, average='micro')
  print('recall score micro for ave selecting', recall_ave)
  precision_ave = metrics.precision_score(y_true, y_pred, average='macro')
  print('precision score macro for ave selecting', precision_ave)
  precision_ave = metrics.precision_score(y_true, y_pred, average='micro')
  print('precision score micro for ave selecting', precision_ave)
  ws.write(0, 2, 'n-NCNN')
  row = 1
  for i in sta_label_set:
      print('label: {}, f1 measure: {}'.format(all_label_inv[i+1], f1_score[i]))
      ws.write(row, 2, f1_score[i])
      row += 1

  print('most selected prediction:')
  acc, y_pred = model.evaluate_select(x_test, y_test, ratio=ratio, method='most', walk=args.walk, batch=256,
                                      except_padding=args.padding, p=args.p, q=args.q)

  y_pred = np.array(y_pred)
  y_true = np.argmax(y_test, axis=-1)
  f1_score_most = metrics.f1_score(y_true, y_pred, average='macro')
  print('f1 score macro for most selecting', f1_score_most)
  f1_score_most = metrics.f1_score(y_true, y_pred, average='micro')
  print('f1 score micro for most selecting', f1_score_most)
  f1_score_most = metrics.f1_score(y_true, y_pred, average='weighted')
  print('f1 score weighted for most selecting', f1_score_most)
  f1_score = metrics.f1_score(y_true, y_pred, average=None)
  recall_most = metrics.recall_score(y_true, y_pred, average='macro')
  print('recall score macro for most selecting', recall_most)
  recall_most = metrics.recall_score(y_true, y_pred, average='micro')
  print('recall score micro for most selecting', recall_most)
  precision_most = metrics.precision_score(y_true, y_pred, average='macro')
  print('precision score macro for most selecting', precision_most)
  precision_most = metrics.precision_score(y_true, y_pred, average='micro')
  print('precision score micro for most selecting', precision_most)
  ws.write(0, 3, 'x-NCNN')
  row = 1
  for i in sta_label_set:
      print('label: {}, f1 measure: {}'.format(all_label_inv[i+1], f1_score[i]))
      ws.write(row, 3, f1_score[i])
      row += 1
  wb.save('D:/data/imbalancedData/results/Results.xls')
