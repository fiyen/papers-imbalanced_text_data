"""
define the ncnn model

author: Dongyang Yan
github: github.com/fiyen
last modification date: 2021/11/04
"""

from tensorflow.keras import Model
from collections import Counter
from tensorflow.keras import activations
from .layers import *
import time
from .random_walk import PathGen
from config import Config
args = Config()


class PcnnNet(Model):
  """
  模型概况，EmbeddingNet-PolarLayerNet-Conv2D-Conv2D-Maxpool-Flatten-Dense-Dense
  """
  def __init__(self, output_dim,
            num_features,
            sequence_length,
            embedding_dim,
            words_list,
            dict_source='dictionary',
            labeled=False,
            word_vectors=None,
            polar=(100, 2, 2),
            polars=None,
            conv_1=(10, (2, 2), (2, 2)),
            conv_2=(20, (4, 4), (4, 4)),
            pooling=(True, True),
            pooling_strides=((2, 2), (2, 2)),
            name='pcnn_model_for_vision',
            **kwargs):
    super(PcnnNet, self).__init__(name=name, **kwargs)
    self.output_dim = output_dim
    self.num_features = num_features
    self.sequence_length = sequence_length
    self.embedding_dim = embedding_dim
    self.words_list = words_list
    self.dict_source = dict_source
    self.labeled = labeled
    self.word_vectors = word_vectors
    self.polar_filters = polar[0]
    self.polar_size = polar[1]
    self.polar_stride = polar[2]
    self.conv_1_filters = conv_1[0]
    self.conv_1_size = conv_1[1]
    self.conv_1_strides = conv_1[2]
    self.conv_2_filters = conv_2[0]
    self.conv_2_size = conv_2[1]
    self.conv_2_strides = conv_2[2]
    self.pooling = pooling
    self.pooling_strides = pooling_strides
    self.deconv_call_ = False

    self.bn_1 = BatchNormalization(trainable=True)
    self.bn_2 = BatchNormalization(trainable=True)
    self.bn_3 = BatchNormalization(trainable=True)
    self.bn_4 = BatchNormalization(trainable=True)

    self.embedding = EmbeddingNet(words_list=words_list, input_dim=num_features, output_dim=embedding_dim,
                    fast_mode=True, input_length=sequence_length, trainable=False,
                    filename=args.vector_file, path=args.vector_dir,
                    preprocessed_vector_path=args.preprocessed_vector_path, name='embedding_layer',
                    dict_source=dict_source, word_vectors=word_vectors, labeled=labeled)
    self.polar = PolarLayerNet(filters=self.polar_filters, kernel_size=self.polar_size,
                  stride=self.polar_stride, polars=polars, use_bias=True, activation='relu',
                  trainable=True, name='polar_layer')
    self.conv_1 = Conv2D(filters=self.conv_1_filters, kernel_size=self.conv_1_size, strides=self.conv_1_strides,
                  padding='VALID', use_bias=True, activation=None, name='1st_conv2d_layer')
    self.conv_2 = Conv2D(filters=self.conv_2_filters, kernel_size=self.conv_2_size, strides=self.conv_2_strides,
                  padding='VALID', use_bias=True, activation=None, name='2rd_conv2d_layer')
    self.conv_3 = Conv2D(filters=self.conv_2_filters, kernel_size=self.conv_2_size, strides=self.conv_2_strides,
                  padding='VALID', use_bias=True, activation=None, name='3rd_conv2d_layer')
    self.pool_1 = MaxPool2D(pool_size=pooling_strides[0], name='1st_max_pooling_layer')
    self.pool_2 = MaxPool2D(pool_size=pooling_strides[1], name='2rd_max_pooling_layer')
    self.flatten = Flatten()
    self.dense_1 = Dense(200, activation='relu', use_bias=True)
    self.dense_2 = Dense(output_dim, activation='softmax')
    self.act = activations.get('relu')
    self.dropout = Dropout(0.5)

  def call(self, inputs, **kwargs):
    """
    :param inputs: (batches, n_random_paths, length_paths)
    :param kwargs:
    :return:
    """
    embeddings = self.embedding(inputs)
    polar_matrix = self.polar(embeddings)
    #polar_matrix = self.bn_1(polar_matrix)
    #polar_matrix = self.act(polar_matrix)
    conv_1_out = self.conv_1(polar_matrix)
    #conv_1_out = self.bn_2(conv_1_out)
    conv_1_out = self.act(conv_1_out)
    pooling_1 = self.pool_1(conv_1_out)
    conv_2_out = self.conv_2(pooling_1)
    #conv_2_out = self.bn_3(conv_2_out)
    conv_2_out = self.act(conv_2_out)
    pooling_2 = self.pool_2(conv_2_out)
    #conv_3_out = self.conv_3(pooling_2)
    #conv_3_out = self.bn_4(conv_3_out)
    #conv_3_out = self.act(conv_3_out)
    #pooling_3 = self.pool_2(conv_3_out)
    flatten = self.flatten(pooling_2)
    dense_1_out = self.dense_1(flatten)
    dense_1_out = self.dropout(dense_1_out)
    dense_2_out = self.dense_2(dense_1_out)
    return dense_2_out

  def evaluate_select(self, x, y, ratio=5, method='most', walk='self_avoiding', batch=256, except_padding='zeros', **kwargs):
    """
    基于数据倍增的评测函数
    :param batch:
    :param walk:
    :param method: 选举的方式，most选择最多的一类，ave将预测概率求平均并求概率最大的一类
    :param ratio:
    :param x:
    :param y:
    :return:
    """
    pred = []
    y = tf.argmax(y, axis=1)
    len_ = y.shape[0]
    #print(len_)
    if batch <= 0:
      y_pred = self.predict_select(x, ratio=ratio, method=method, walk=walk, except_padding=except_padding, **kwargs)
      pred += [x for x in np.array(y_pred)]
      tof = [1 if y[i] == y_pred[i] else 0 for i in range(len_)]
      accuracy = sum(tof) / len_
    else:
      batch_num = int(tf.math.ceil(len_ / batch))
      accuracy = 0
      cumulate_len = 0
      cumulate_time = 0
      for i in range(batch_num):
        start = time.time()
        x_ = x[i*batch:(i+1)*batch]
        y_ = y[i*batch:(i+1)*batch]
        if i < batch_num-1:
          len__ = batch
        else:
          len__ = len_ % batch
        y_pred = self.predict_select(x_, ratio=ratio, method=method, walk=walk, except_padding=except_padding, **kwargs)
        pred += [x for x in np.array(y_pred)]
        tof = [1 if y_[i] == y_pred[i] else 0 for i in range(len__)]
        accuracy = (accuracy * cumulate_len + sum(tof)) / (cumulate_len + len__)
        cumulate_len = cumulate_len + len__
        if i < batch_num-1:
          cumulate_time = cumulate_time + time.time() - start
          print('{}/{} - ETA: {:.0f}s - accuracy: {:.4f}'.
              format(str(cumulate_len).rjust(len(str(len_))), len_,
                    (len_ - cumulate_len) * cumulate_time / cumulate_len, accuracy))
        else:
          cumulate_time = cumulate_time + time.time() - start
          print('{}/{} - {:.0f}s - accuracy: {:.4f}'.
              format(str(cumulate_len).rjust(len(str(len_))), len_, cumulate_time, accuracy))

    return accuracy, pred

  def predict_select(self, x, y=None, ratio=5, method='most', walk='self_avoiding', except_padding='zeros', **kwargs):
    """
    基于数据倍增的预测
    :param x:
    :param y: not used
    :param ratio:
    :param method: 选举的方式，most选择最多的一类，ave将预测概率求平均并求概率最大的一类
    :param kwargs:
    :param walk:
    :return:
    """
    get_path = PathGen(self.polar_filters, self.sequence_length, walk=walk, expansion_ratio=ratio, except_padding=except_padding, **kwargs)
    if isinstance(x, list):
      len_ = len(x)
    else:
      len_ = x.shape[0]
    x, _ = get_path(x)
    x = [x[i*len_:(i+1)*len_] for i in range(ratio)]
    if method == 'most':
      pred = [tf.argmax(self.predict(x_), axis=1) for x_ in x]
      #print(pred)
      pred = tf.stack(pred, axis=1)
      #print(pred)
      pred = tf.unstack(pred, axis=0)
      #print(pred)
      pred = [Counter([int(p_) for p_ in list(p)]).most_common(1)[0][0] for p in pred]
      #print(pred)
    else:
      pred = [self.predict(x_) for x_ in x]
      pred = tf.stack(pred, axis=-1)
      pred = tf.reduce_sum(pred, axis=-1)
      pred = tf.argmax(pred, axis=-1)
      #print(pred)
    return pred
