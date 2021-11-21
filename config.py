"""
Set the configuration of the model and datasets.

author: Dongyang Yan
github: github.com/fiyen
last modification date: 2021/11/04
"""


class Config:
  # datasets related
  datasets_dir = "D:/data/imbalancedData/datasets"
  file_name = 'webkb'  # @param ['r52','r8','20ng','amazon-reviews','webkb']
  file_form = 'stemmed'  # @param ['no-stop','stemmed','all-terms']
  file_suffix = '.txt'  # @param ['.txt','.vec','.bin']
  fname = 'D:/data/imbalancedData/w2v/w2v'

  # embeddings related
  embedding_dimension = 300  # @param {type:'integer'}
  word2vec_training_epochs = 100  # @param {type:'integer'}
  ## save or load w2v file
  load_option = False  # @param {type:'boolean'}
  save_option = False  # @param {type:'boolean'}
  dict_source = 'dictionary'  # @param ['dictionary', 'trained']
  # 当dict_source为Trained时需标注是否labeled。如果为False，则查询单词符号，否则查询单词编码。
  labeled = False  # @param {type:'boolean'}
  # whether to activate parallel computing
  threads4gen = 6

  ## pretrained word vectors
  vector_file = 'vectors.txt'
  vector_dir = 'D:/data/imbalancedData/w2v'
  preprocessed_vector_path = 'C:/test_output/cache/en'

  # random path generation
  ## see doc of PathGen for the description of the following params
  walk = 'node2vec'  # @param ['self_avoiding', 'normal', 'node2vec', 'weighted', 'weighted_reverse', 'smooth', 'smooth_reverse']
  p = 2.0  # @param{type:""}
  q = 1.0  # @param{type:""}
  sample_cross = True  # @param{type:"boolean"}
  undersampling = False  # @param{type:'boolean'}
  padding = 'zeros'  # @param ['zeros', 'random']
  cut_off_num = 400  # @param {type:'integer'}
  expansion_ratio = 1  # @param {type:'integer'}
  adaptive_expansion = True  # @param {type:'boolean'}
  ## random walk length
  sequence_length = 400  # @param {type:'integer'}

  # NCNN network related
  ## polar cores
  polar_size = 40  # @param {type:'integer'}
  ## convolution core 1
  conv_1 = (60, (3, 3), (2, 1))
  ## convolution core 2
  conv_2 = (40, (2, 2), (1, 1))
  ## whether to take pooling after each convolution
  pooling = (True, True)
  pooling_strides = ((2, 1), (2, 1))

  # training
  epochs = 10
  validation_split = 0.1
  batch_size = 256
