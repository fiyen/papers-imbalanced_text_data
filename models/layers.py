"""
define the layers of NCNN.

author: Dongyang Yan
github: github.com/fiyen
last modification date: 2021/11/04
"""
from tensorflow.python.ops.init_ops_v2 import Initializer
from tensorflow.python.framework import dtypes, constant_op
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
import math
from .word_embedding import EnVectorizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# EmbeddingInitializer. To assist the embedding of the Embedding Layer when the embeddings are pre-trained.
class EmbeddingInitializer(Initializer):
  """
  初始化embedding层的权重值
  """
  def __init__(self, value=0):
      if not (np.isscalar(value) or isinstance(value, (list, tuple, np.ndarray))):
          raise TypeError(
              "Invalid type for initial value: %s (expected Python scalar, list or "
              "tuple of values, or numpy.ndarray)." % type(value))
      self.value = value

  def __call__(self, shape=None, dtype=dtypes.float32):
      """
      返回value,shape不用填
      :param shape:
      :param dtype:
      :return: self.value
      """
      shape = self.value.shape
      if dtype is not None:
          dtype = dtypes.as_dtype(dtype)
      return constant_op.constant(
          self.value, dtype=dtype, shape=shape)

  def get_config(self):
      return {"value": self.value}


# EmbeddingNet Layer. The first layer in pcnn model.
from tensorflow.keras import layers
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops


class Embedding(layers.Layer):
  """
  重写Embedding类，要求Embedding时对每个词附以已经经过训练得到的向量，并且根据情况选择是否可训练。
  """
  def __init__(self,
        words_list,  # 按编码顺序排列的词列表，用于查询词向量
        input_dim,
        output_dim,
        dict_source='dictionary',  # 查询字典的来源，dictionary表示查询训练好的向量文件，trained表示由训练得到
        labeled=False,  # 当dict_source为Trained时需标注是否labeled。如果为False，则查询单词符号，否则查询单词编码。
        filename=None,  # 词向量文件的名称
        path=None,  # 词向量文件的位置
        fast_mode=True,  # 搜索过程是否使用并行模式
        preprocessed_vector_path='C:/test_output',  # 使用并行模式时字典分文件的地址
        word_vectors=None,  # 当dict_source=='trained'时必须给定，基于gensim的wv
        mask_zero=False,
        input_length=None,
        trainable=False,
        **kwargs):
    super(Embedding, self).__init__(
                    #input_dim=input_dim,
                    #output_dim=output_dim,
                    #mask_zero=mask_zero,
                    #input_length=input_length,
                    **kwargs)
    if len(words_list) > input_dim:
      raise ValueError("The length of words_list should not larger than input_dim.")
    if not isinstance(words_list, list):
      raise TypeError("words_list must be list type")
    self.words_list = words_list
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.dict_source = dict_source
    self.labeled = labeled
    self.word_vectors = word_vectors
    self.filename = filename
    self.path = path
    self.fast_mode = fast_mode
    self.preprocessed_vector_path = preprocessed_vector_path
    self.mask_zero = mask_zero
    self.input_length = input_length
    self.trainable = trainable
    if self.dict_source == 'trained' and self.word_vectors is None:
      raise ValueError('word_vectors should be given if dict_source is "trained".')

  def build(self, input_shape):
    if self.dict_source == 'dictionary':
      my_vectorizer = EnVectorizer(fast_mode=self.fast_mode)
      my_vectorizer.get_path(self.filename, self.path, self.preprocessed_vector_path)
      embedded_words = my_vectorizer.lookup(self.words_list)
    else:
      if self.labeled:
        embedded_words = [self.word_vectors[str(w)].tolist() for w in range(len(self.words_list))]
      else:
        embedded_words = [self.word_vectors[str(w)].tolist() for w in self.words_list[1:]]
        embedded_words = [np.zeros((self.output_dim,)).tolist()] + embedded_words
    embedded_words = embedded_words + self._free_padding(self.input_dim - len(embedded_words))
    # print("embedded_words:", embedded_words)
    init = EmbeddingInitializer(value=np.array(embedded_words))
    if context.executing_eagerly() and context.context().num_gpus():
      with ops.device('cpu:0'):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            name='embeddings',
            trainable=self.trainable,
            initializer=init,
            )
    else:
      self.embeddings = self.add_weight(
        shape=(self.input_dim, self.output_dim),
        name='embeddings',
        trainable=self.trainable,
        initializer=init,
        )
    self.built = True

  def call(self, inputs):
    dtype = backend.dtype(inputs)
    if dtype != 'int32' and dtype != 'int64':
      inputs = tf.cast(inputs, 'int32')
    out = embedding_ops.embedding_lookup(self.embeddings, inputs)
    return out

  def _free_padding(self, length):
    """
    随机产生并返回一个维度为(length, output_dim)的向量
    :param length:
    :return: vectors
    """
    return np.random.standard_normal((length, self.output_dim)).tolist()


class EmbeddingNet(Embedding):
  """
  文本网络随机游走生成的游走路径为一个二维的数组，其格式为（batches, n_random_paths, path_length). 可以视为是n_random_paths个
  （batches, length=path_length)的普通文本样本输入，则对于二维随机游走路径的Embedding是n_random_paths次普通文本的Embedding。
  """

  def __init__(self,
        words_list,  # 按编码顺序排列的词列表，用于查询词向量
        input_dim,
        output_dim,
        filename=None,  # 词向量文件的名称
        path=None,  # 词向量文件的位置
        fast_mode=True,  # 搜索过程是否使用并行模式
        preprocessed_vector_path='C:/test_output',  # 使用并行模式时字典分文件的地址
        mask_zero=True,
        input_length=None,
        trainable=False,
        **kwargs):
    super(EmbeddingNet, self).__init__(words_list=words_list,
                      input_dim=input_dim,
                      output_dim=output_dim,
                      filename=filename,
                      path=path,
                      fast_mode=fast_mode,
                      preprocessed_vector_path=preprocessed_vector_path,
                      mask_zero=mask_zero,
                      input_length=input_length,
                      trainable=trainable,
                      **kwargs)
    if len(words_list) > input_dim:
      raise ValueError("The length of words_list should not larger than input_dim.")
    if not isinstance(words_list, list):
      raise TypeError("words_list must be list type")
    self.words_list = words_list
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.filename = filename
    self.path = path
    self.fast_mode = fast_mode
    self.preprocessed_vector_path = preprocessed_vector_path
    self.mask_zero = mask_zero
    self.input_length = input_length
    self.trainable = trainable

  def call(self, inputs):
    """
    :param inputs: （batches, n_random_paths, path_length)
    :return:
    """
    dtype = backend.dtype(inputs)
    if dtype != 'int32' and dtype != 'int64':
        inputs = tf.cast(inputs, 'int32')
    x = tf.unstack(inputs, axis=1)
    #print(x[0].shape)
    lookup = embedding_ops.embedding_lookup
    out = [lookup(self.embeddings, x_) for x_ in x]
    out = tf.stack(out, axis=1)  # 将n_random_paths维移到最后，与pcnn_layer对应
    #print(out.shape)
    out = tf.transpose(out, perm=[0, 2, 3, 1])
    #print(out.shape)
    return out


# PolarLayerNet. The layer follows the EmbeddingNet Layer to polarize the embeddings to 2D data.
from tensorflow.keras import activations


class PolarLayer(layers.Layer):
  """
  连接Embedding层的偏振层，将Embedding后的输入信号转换成二维信号，其中行表示不同filter，列表示卷积后的文本信息
  input_shape: (width, dimension), width,文本长度，dimension，词向量的维数
  kernel_size: int, kernel的大小
  stride: int, 卷积的步长
  polars: 词向量，长度等于filters，当polar=None时，该层的卷积核随机赋值，是否可训练取决于trainable，当polar不是None时，该层
  的卷积核的初始值由polar给定，并且trainable=False.目的是克服初始值对优化效果的影响。格式为Tensor,（filters, dimension)
  padding: 填充方式，注意，COVER的方式比VALID的方式卷积后的结果多1，或者相等(当(width - kernel_size) % stride == 0时）
  use_bias: True or False, 是否用偏执项
  activation: 激活选项，默认为None
  """
  def __init__(self, filters,
        kernel_size,
        stride=1,
        polars=None,
        use_bias=False,
        activation=None,
        padding='VALID',
        trainable=True,
        **kwargs
        ):
    super(PolarLayer, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.stride = stride
    self.polars = polars
    self.padding = padding
    self.use_bias = use_bias
    self.trainable = trainable
    self.activation = activations.get(activation)

  def build(self, input_shape):
    """
    input_shape (batch, width, dimension)
    """
    # print('input_shape', input_shape)
    dim = input_shape[2]
    width = input_shape[1]
    if self.padding == 'VALID':
      filter_size = math.floor((width - self.kernel_size) / self.stride) + 1
    elif self.padding == 'SAME':
      filter_size = width
    elif self.padding == 'COVER':
      filter_size = math.ceil((width - self.kernel_size) / self.stride) + 1
    else:
      raise TypeError("No padding type is named {}! Only support type (VALID, "
                  "SAME, COVER)".format(self.padding))
    kernel_shape = (self.kernel_size, dim, self.filters)
    if self.polars is None:
      self.polar_kernel = self.add_weight(name='polar_kernel',
                        shape=kernel_shape,
                        initializer=tf.keras.initializers.glorot_uniform,
                        trainable=self.trainable)
    else:
      if self.polars.shape[0] != self.filters:
        raise ValueError("The shape of polars doesn't match the filters. shape:{}, filters:{}"
                    .format(self.polars.shape, self.filters))
      else:
        self.trainable = False
        new_polars = []
        for i in range(self.kernel_size):
            new_polars.append(self.polars)
        new_polars = tf.stack(new_polars, axis=0)
        new_polars = tf.transpose(new_polars, perm=[0, 2, 1])
        init = EmbeddingInitializer(value=new_polars.numpy())
        self.polar_kernel = self.add_weight(name='polar_kernel',
                          shape=kernel_shape,
                          initializer=init,
                          trainable=self.trainable)
    if self.use_bias:
      self.polar_bias = self.add_weight(name='polar_bias',
                        shape=(self.filters,),
                        initializer=tf.keras.initializers.zeros,
                        trainable=self.trainable)
    self.built = True

  def call(self, inputs, **kwargs):
    """
    inputs: (batch, width, dimension)
    """
    self.inputs_shape = inputs.shape
    padding_f = self.padding_get(self.padding)
    inputs = padding_f(inputs)
    inputs = tf.expand_dims(inputs, axis=-1)  # (batch, width, dimension, filter=1)
    x = tf.unstack(inputs, axis=1)
    x = tf.stack(
      [x[i:i + self.kernel_size] for i in range(0, len(x) - self.kernel_size + 1, self.stride)]
          )  # (num, width, batch, dimension, filter=1)
    kernel = tf.expand_dims(self.polar_kernel, axis=1)  # (width, batch=1, dimension, filter)
    len_x = x.shape[0]
    x = tf.stack([tf.reduce_sum(x[i] * kernel, [0, 2]) for i in range(len_x)])  # (num, batch, filter) num=filter
    out = tf.transpose(x, perm=[1, 0, 2])
    if self.use_bias:
      bias = tf.expand_dims(self.polar_bias, 0)
      bias = tf.expand_dims(bias, 0)
      out = out + bias
    out = tf.expand_dims(out, axis=-1)
    out = self.activation(out)
    # 归一化
    out = (out - tf.reduce_min(out, [1,2,3], keepdims=True)) / \
          (tf.reduce_max(out, [1,2,3], keepdims=True) - tf.reduce_min(out, [1,2,3], keepdims=True))
    return out

  def padding_same(self, inputs, padded_dim=1):
    """
    inputs的shape为(batches, width, dimension),对width进行padding, 以使call的输出与输入长度相同
    """
    inputs_length = self.inputs_shape[padded_dim]
    padded_length = (inputs_length - 1) * self.stride + self.kernel_size
    dif_length = padded_length - inputs_length
    padding_length_start = math.floor(dif_length / 2)
    padding_length_end = dif_length - padding_length_start
    padding = np.zeros((len(self.inputs_shape), 2), dtype=int)
    padding[padded_dim][0] = padding_length_start
    padding[padded_dim][1] = padding_length_end
    return tf.pad(inputs, paddings=padding)

  def padding_valid(self, inputs, padded_dim=None):
    return inputs

  def padding_cover(self, inputs, padded_dim=1):
    inputs_length = self.inputs_shape[padded_dim]
    if (inputs_length - self.kernel_size) % self.stride == 0:
      return inputs
    else:
      padding_length_end = self.stride - ((inputs_length - self.kernel_size) % self.stride)
      padding = np.zeros((len(self.inputs_shape), 2), dtype=int)
      padding[padded_dim][1] = padding_length_end
      return tf.pad(inputs, paddings=padding)

  def padding_get(self, padding):
    if padding is 'SAME':
      return self.padding_same
    elif padding is 'VALID':
      return self.padding_valid
    elif padding is 'COVER':
      return self.padding_cover
    else:
      raise TypeError("Only support padding (SAME, VALID, COVER), No padding type {}".format(padding))


class PolarLayerNet(PolarLayer):
  """
  与PolarLayer基本一致，不同点在于，PolarLayer是将每个一维文本数据卷积filters次形成filters层的输出，而PolarLayerNet是将有
  filters行的二维随机游走数据与filters层的polar_kernel做卷积操作，每一个filters层的polar_kernel做卷积的对象都不相同。
  """
  def __init__(self, filters,
        kernel_size,
        stride=1,
        polars=None,
        use_bias=False,
        activation=None,
        padding='VALID',
        trainable=True,
        **kwargs
        ):
    super(PolarLayerNet, self).__init__(filters=filters,
                      kernel_size=kernel_size,
                      stride=stride,
                      polars=polars,
                      use_bias=use_bias,
                      activation=activation,
                      padding=padding,
                      trainable=trainable,
                      **kwargs)
    print("Note: Please verify that filters is equal to number of random paths. If some Errors are raised up, first"
          "check the above note.")

  def call(self, inputs, **kwargs):
    """
    inputs: (batch, width, dimension, filters=n_random_paths)
    """
    self.inputs_shape = inputs.shape
    #print('inputs_shape', self.input_shape)
    padding_f = self.padding_get(self.padding)
    inputs = padding_f(inputs)
    x = tf.unstack(inputs, axis=1)
    x = tf.stack(
        [x[i:i + self.kernel_size] for i in range(0, len(x) - self.kernel_size + 1, self.stride)]
                )  # (num, width, batch, dimension, filter=1)
    kernel = tf.expand_dims(self.polar_kernel, axis=1)  # (width, batch=1, dimension, filter)
    #print('kernel size', kernel.shape)
    len_x = x.shape[0]
    #print('x_shape', x.shape)
    x = tf.stack([tf.reduce_sum(x[i] * kernel, [0, 2]) for i in range(len_x)])  # (num, batch, filter)
    out = tf.transpose(x, perm=[1, 0, 2])
    if self.use_bias:
        bias = tf.expand_dims(self.polar_bias, 0)
        bias = tf.expand_dims(bias, 0)
        out = out + bias
    out = tf.expand_dims(out, axis=-1)
    out = self.activation(out)
    # 归一化
    out = (out - tf.reduce_min(out, [1,2,3], keepdims=True)) / \
          (tf.reduce_max(out, [1,2,3], keepdims=True) - tf.reduce_min(out, [1,2,3], keepdims=True))
    return out


# Conv2D Layer, MaxPooling Layer, Flatten Layer, and Dense Layer
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Activation