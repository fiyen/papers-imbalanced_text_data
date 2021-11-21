"""
Train or lookup the word vectors.

author: Dongyang Yan
github: github.com/fiyen
last modification date: 2021/11/04
"""

from gensim import models
import time
import numpy as np
import os
import shutil
from multiprocessing import pool, cpu_count


# WordEmbedding. A word2vec startegy for pre-training word embedding.
class WordEmbedding:
  """
  对数据进行word2vec训练，返回训练之后的词向量
  size: 词向量的维度
  epochs: 训练的周期数
  save: 是否储存模型
  fname: 储存模型的位置和名字
  """
  def __init__(self, size=300, epochs=100, min_count=5, save=True, load=False, fname='w2v'):
    self.size = size
    self.epochs = epochs
    self.min_count = min_count
    self.save = save
    self._load = load
    self.fname = fname
    self.fited = False
    self.loaded = False
    self.w2v = None

  def fit_train(self, data):
    """
    data为list或array
    :param data:
    :return:
    """
    if self._load:
      print("Word2vector start......")
      start = time.time()
      self.load(fname=self.fname)
      print("Word2vector finished. Calling load function. Time costs: {:.2f}s.".format(time.time() - start))
      return self
    else:
      if not isinstance(data, (list, np.ndarray)):
        raise TypeError("train_data and valid_data should all be type list or np.ndarray.")
      try:
        data = data.tolist()
      except AttributeError:
        pass
      try:
        data = [line.split() for line in data]
      except AttributeError:
        pass
      print("Word2vector start......")
      start = time.time()
      self.w2v = models.Word2Vec(vector_size=self.size, min_count=self.min_count)
      self.w2v.build_vocab(data)
      self.w2v.train(data, total_examples=self.w2v.corpus_total_words, epochs=self.epochs)
      if self.save:
        self.w2v.save(self.fname)
      print("Word2vector finished. Time costs: {:.2f}s.".format(time.time() - start))
      return self

  def get_wv(self):
    """
    得到训练好的词向量
    :return:
    """
    if not self.fited and not self.loaded:
      raise NotImplementedError("Function fit or load should be called first.")
    return self.w2v.wv

  def load(self, fname=None):
    """
    读取已经训练好的模型
    :param fname:
    :return:
    """
    if fname is None:
      fname = self.fname
    self.w2v = models.Word2Vec.load(fname)
    self.loaded = True
    return self


# EnVectorizer. To lookup the embeddings from a dictionary.
class EnVectorizer:
  """
  映射英文单词的词向量
  """

  def __init__(self, fast_mode=True, need_pro=False):
    self.fast_mode = fast_mode  # 是否开启快速查找模式，默认开启。如果开启快速查找，则会先将word_vector文件按照split_label分成子文件,然后并行查找子文件
    self.filename = 'glove.6B.50d.txt'
    self.path = 'D:/onedrive/work/word_vector'
    self.encoding = 'utf-8'
    self.preprocessed_vector_path = 'preprocessed_word_vector'  # fast_mode中的子文件保存路径
    self.need_pro = need_pro  # 是否需要预处理字典文件,默认为False
    self.filehead = None  # 打开文件的头地址
    self.split_label = {}  # 读取字典分文件名，每个符号代表一个分文件，其中一定有“+”
    for x in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'+":  # +表示else，除了前边的所有符号以外的所有符号
      if x.lower() == x:
        self.split_label[x] = x
      else:
        self.split_label[x] = x + '_'

  def get_path(self, filename=None, path=None, pre_path=None, encoding='utf-8'):
    """
    更改默认的词向量文件名和文件地址
    :param filename: 文件名
    :param path: 文件地址
    :param pre_path: 子文件地址
    :param encoding: 文件编码格式
    :return:
    """
    if filename is not None:
      self.filename = filename
    if path is not None:
      self.path = path
    self.get_encoding(encoding)
    if pre_path is not None:
      self.preprocessed_vector_path = pre_path

  def get_split_label(self, split_label):
    """
    更改split_label
    """
    self.split_label = split_label

  def get_encoding(self, encoding):
    """
    更改读入和写出文件的编码格式
    """
    self.encoding = encoding

  def _build(self):
    """
    预处理部分，读取文件中的词向量并返回词向量字典
    :return:
    """
    if not self.fast_mode:
      path = self.path + '/' + self.filename
      self.filehead = open(path, 'r', encoding=self.encoding)
    else:
      self.filehead = {}
      for label in self.split_label.keys():
        self.filehead[label] = self.preprocessed_vector_path + '/' + self.split_label[label]

  def _process(self, lines):
    """
    对文件读取的lines处理得到words和vector
    :param lines:
    :return: {words， vector}
    """
    out = {}
    lines = [line.split() for line in lines]
    for line in lines:
      try:
        out[line[0]] = [float(num) for num in line[1:]]
      except ValueError:
        continue
    return out

  def _free_padding(self, vectors):
    """
    对没有进行向量映射的单词赋予随机的向量值, 如果self.skip=True, 第一行填充为0，为占位符
    """
    length = 0
    num_rand = 0
    num_all = len(vectors)
    for i in vectors:
      if i:
        length = len(i)
    if length == 0:
      raise ValueError("It seems that all vectors are not mapped, this will cause problem for the next process. ")
    for index, vector in enumerate(vectors):
      if not vector:
        num_rand += 1
        vectors[index] = np.random.standard_normal((length,)).tolist()
    print("There are {} vectors, including {} mapped vectors and {} randomized vectors"
          .format(num_all, num_all - num_rand, num_rand))
    if self.skip:
      vectors[0] = np.zeros((length,)).tolist()
      print("The 'skip' is True, it seems that the first label is a padding label, so it is initialized to zeros.")
    return vectors

  # @ time_decorator
  def lookup(self, words, batch=1000, epochs=10000, skip=True, threads=None):
    """
    查找单词的词向量，如果没有找到，则词向量被随机赋予一个向量
    :param words: 词集, 必须是list格式
    :param batch: 每个周期读取文件的行数
    :param epochs: 查找周期数
    :param skip:  是否跳过第一个词（第一个词可能是占位词）
    :param threads: 并行线程数，只有fast_mode下才有效，如果是None，则线程数等于cpu的线程数
    :return:
    """
    self.skip = skip
    if not isinstance(words, list):
      raise TypeError("words must be list type.")
    self.words_dict = {word: index for index, word in enumerate(words)}
    if not self.fast_mode:
      words = set(words)
      vectors = [[] for i in range(len(words))]
      self._build()
      left_num = 0
      flag = 0
      if skip:
        left_num = 1
      for epoch in range(epochs):
        lines = [[] for i in range(batch)]
        for bat in range(batch):
          line = self.filehead.readline()
          if line:
            lines[bat] = line
          else:
            lines = lines[:bat]
            flag = 1
            print("Epoch: {}, The End of File, So Break.".format(epoch))
            break
        vector_ = self._process(lines)
        """for word in words:
            try:
                vectors[self.words[word]] = vector_[word]
            except KeyError:
                left_words.append(word)"""
        words_int = words.intersection(vector_.keys())
        words.difference_update(words_int)
        for word in words_int:
          vectors[self.words_dict[word]] = vector_[word]
        if len(words) <= left_num:
          print("Epoch: {}, All Words Are Mapped, So Break.".format(epoch))
          print("All words are mapped.")
          return vectors
        if flag == 1:
          vectors = self._free_padding(vectors)
          return vectors
        if ((epoch + 1) * batch) % 100000 == 0:
          print("Epoch: {}/{}, Complete.".format(epoch + 1, epochs))
      print("Epoch is Enough, So Return.")
      vectors = self._free_padding(vectors)
      return vectors
    else:  # fast_mode模式，用并行运算，分文件进行查找
      # 首先，对待查询的words，分成以split_label中各符号为首字母的单词字典
      # 其中， 字典的value值也是一个字典，key为单词，value为该单词在words中的index
      prepro = PreprocessVector(need_pro=self.need_pro)
      prepro.save_path(self.preprocessed_vector_path)
      prepro.get_path(filename=self.filename, path=self.path)
      prepro.subfile_name(self.split_label)
      prepro.process()
      vectors = [[] for i in range(len(words))]
      words = self._split()
      self._build()
      if threads is None:
        threads = cpu_count()
      p = pool.Pool(threads)
      labels = [label for label in self.split_label.keys()]
      words_dicts = [words[label] for label in labels]
      batch_bag = [batch for i in range(len(labels))]
      epochs_bag = [epochs for i in range(len(labels))]
      skip_bag = [skip for i in range(len(labels))]
      vectors_dicts = p.map(self._swap, zip(labels, words_dicts, batch_bag, epochs_bag, skip_bag))
      p.close()
      for vectors_dict in vectors_dicts:
        for key in vectors_dict.keys():
          vectors[key] = vectors_dict[key]
      return self._free_padding(vectors)

  def _swap(self, args):
    return self._fast_lookup(*args)

  def _fast_lookup(self, label, words_dict, batch=100, epochs=10000, skip=True):
    """
    快速查找模式中的并行的进程
    words_dict: 字典， keys(): 单词， values(): 单词在原words列表中的index
    """
    # print(label)
    vectors = {}
    file = open(self.filehead[label], 'r', encoding=self.encoding)
    words = set(words_dict.keys())
    left_num = 0
    if skip and label == '+':
      left_num = 1
    flag = 0
    for epoch in range(epochs):
      lines = [[] for i in range(batch)]
      for bat in range(batch):
        line = file.readline()
        if line:
          lines[bat] = line
        else:
          lines = lines[:bat]
          flag = 1
          print("The End of File, So Break. Split label is \033[1m{}\033[0m, ".format(label))
          break
      # print(lines)
      vector_ = self._process(lines)
      words_int = words.intersection(vector_.keys())
      words.difference_update(words_int)
      for word in words_int:
        vectors[words_dict[word]] = vector_[word]
      if len(words) <= left_num:
        print("All Words Are Mapped, So Break. Split label is \033[1m{}\033[0m, ".format(label))
        return vectors
      if flag == 1:
        return vectors
    print("Epoch is Enough, So Return. Split label is \033[1m{}\033[0m, ".format(label))
    return vectors

  def _split(self):
    """
    将words按照split_label分成一个字典，其格式如下
    1. 该字典的key值是split_label中各符号，value值是以该符号为首字母的单词字典。其中“+”表示不以split_label中除“+”以外任意字符开头
    的单词
    2. 字典的value值也是一个字典，key为单词，value为该单词在words中的index
    :param words: 待查词集
    :return: 分类后的字典
    """
    words_dict = {label: {} for label in self.split_label.keys()}
    split_label = [label for label in self.split_label.keys() if label is not '+']
    for word in self.words_dict.keys():
      flag = 1  # 用来标记word是否不以split_label中除“+”以外任意字符开头
      for label in split_label:
        if word[0] == label:
          flag = 0
          words_dict[label][word] = self.words_dict[word]
          break
      if flag == 1:
        words_dict['+'][word] = self.words_dict[word]
    return words_dict


# PreprocessVector. To split the dictionary to categories for a speedup in looking up.
class PreprocessVector:
  """
  将容量大的word vector按首字母切分成小的文件，以便于后续查找
  need_pro, 是否需要忽略已经切分好的文件重新生成文件
  """
  def __init__(self, need_pro=False):
    self.need_pro = need_pro
    self.filename = 'glove.6B.50d.txt'
    self.path = 'D:/onedrive/work/word_vector'
    self.subfile_path = 'preprocessed_word_vector'
    self.valid = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'"
    self.dict = {}
    for x in self.valid + '+':  # +表示else，除了前边的所有符号以外的所有符号
      if x.lower() == x:
        self.dict[x] = x
      else:
        self.dict[x] = x + '_'
    self.filehead = None  # 打开文件的头地址

  def get_path(self, filename, path=None):
    """
    更改默认的词向量文件名和文件地址
    :param filename: 文件名
    :param path: 文件地址
    :return:
    """
    self.filename = filename
    if path is not None:
      self.path = path

  def save_path(self, path):
    """
    更改默认的切分文件存放地址
    """
    self.subfile_path = path

  def subfile_name(self, dict):
    """
    设定dict, {查询符号: 文件名}
    注意，dict中必须存在+，表示除了dict中符号之外的其他符号
    """
    self.dict = dict
    self.valid = [x for x in dict.keys() if x is not '+']

  def _search(self):
    """
    寻找subfile_path下是否包含所有需要的子文件
    """
    for filename in self.dict.values():
      if not os.path.exists(self.subfile_path + '/' + filename):
        return False
    return True

  def process(self,
              batch=10000,
              encoding='utf-8',
              sorted=False,
              threads=None,
              split_label='&cut&',
              end_label='000000000',
              remove=True):
    """
    进行文件切分
    :param batch: 每次读入的文件行数，取适当的值可以最大化利用cpu性能，默认1000，过大会造成cpu空置，过小会浪费硬盘吞吐能力
                另外，batch小的时候，可以基本保持源文件中每行的先后顺序, 但是会增加内存的开销
    :param encoding: 文件的编码格式
    :param sorted: 是否需要对切分后的文件重新排序，以使子文件中每行的顺序与源文件相同，不建议排序，因为耗时较长，且需要重排的
                数据不多，切分后的数据已与源文件基本相同，相差不大
    :param threads: 线程数，不能大于计算机cpu的线程数，否则反而会拖慢速度。如果不设置，则默认等于cpu的线程数
    :param split_label: 当sorted为True时使用，split时切割的标志
    :param end_label: line首尾标志，用来标记正确处理的line
    :param remove: 用于确定是否删除临时文件，仅用于sorted为True时
    注意：1.如果sorted为True，则会先将切分后的子文件每行之前加上其在源文件中的排序，然后调用sort函数进行重新排序
         2.函数处理速度受硬盘吞吐量的影响，所以建议先用save_path函数将切分文件存在固态硬盘，然后调用copy函数将文件拷回目标文件夹
         3.并行过程会随机出现一种错误，写入的line会以随机长度被写入两行，导致后续处理出错，生成的数据不能直接使用，必须处理掉错误
         生成的那一部分line，这个错误只有用pool.Pool().apply时才可避免，但是apply的速度是最慢的。
         4.引入end_label来标记每一行，只有包含头尾都包含end_label的line才会被认为是正确的格式，进而被处理
    """
    self.batch = batch
    self.sorted = sorted
    self.encoding = encoding
    self.split_label = split_label
    self.end_label = end_label
    if threads is None:
      threads = cpu_count()
    start_time = time.time()
    print("Start Preprocessing Vectors: This line only shows when 'process' in class PreprocessVector is called.")
    if self.need_pro or not self._search():
      if not os.path.exists(self.subfile_path):
        os.makedirs(self.subfile_path)
      if self.need_pro:
        dirs = os.listdir(self.subfile_path)
        for i in dirs:
          os.remove(self.subfile_path + '/' + i)
      file = open(self.path + '/' + self.filename, 'r', encoding=encoding)
      p = pool.Pool(threads)
      # 为了避免第一行是整个文件的注释，这里做一些处理
      tof = file.readline()
      if len(tof.split()) > 10:  # 10是个随机给的值，因为如果是正常的格式，肯定大于10
        file.seek(0)
      tem = [[file.readline() for x in range(self.batch)] for y in range(threads)]
      subfiles = [[line.strip() for line in lines if line] for lines in tem if lines[0]]
      start = 0
      while len(subfiles) > 0:
        starts = []
        for lines in subfiles:
          starts.append(start)
          start = start + len(lines)
        for f, s in zip(subfiles, starts):
          p.apply(self._process, args=(f, s))
        # p.map_async(self._wrap_p, zip(subfiles, starts))
        tem = [[file.readline() for x in range(self.batch)] for y in range(threads)]
        subfiles = [[line.strip() for line in lines if line] for lines in tem if lines[0]]
      p.close()
      # p.join()
      file.close()
      print("Preprocessing Operation is Completed. Cost Time is {:.2f}s".format(time.time() - start_time))
    if sorted is True:
      self.sort(encoding=encoding, remove=remove)

  def _wrap_p(self, args):
    return self._process(*args)

  def _process(self, file, start):
    """
    并行处理的并行进程，双参数
    :param file:
    :param start: file的首行在源文件的位置
    :return:
    """
    encoding = self.encoding
    if len(file) == 0:
      return 0
    obj = {}
    for key in self.dict.keys():
      obj[key] = open(self.subfile_path + '/' + self.dict[key], 'a', encoding=encoding)
    if self.sorted is False:
      # 写入文件格式，line
      for index, line in enumerate(file):
        if line[0] in self.valid:
          obj[line[0]].write(line + '\n')
        else:
          obj['+'].write(line + '\n')
    else:
      # 写入文件格式，end_label+index+split_label+line+end_label
      for index, line in enumerate(file):
        if line[0] in self.valid:
          obj[line[0]].write(self.end_label + str(start + index) + self.split_label + line + self.end_label + '\n')
        else:
          obj['+'].write(self.end_label + str(start + index) + self.split_label + line + self.end_label + '\n')
    for key in self.dict.keys():
      obj[key].close()

  def copy(self, target_path, source_path=None):
    """
    将处理后的所有文件复制到target_path中, source_path提供了可选的待复制文件地址，默认情况下，source_path=subfile_path
    """
    if source_path is None:
      source_path = self.subfile_path
    start_time = time.time()
    file_list = os.listdir(self.subfile_path)
    if len(file_list) == 0:
      print("It seems that the target_path has no files, maybe function 'process' is not called.")
      return 0
    for file in file_list:
      shutil.copy(source_path + '/' + file, target_path)
    print("Copy Operation is Completed, The Files are Copied to {}. Cost Time is {:.2f}s"
          .format(target_path, time.time() - start_time))

  def sort(self,
           target_path=None,
           lines_per_tempfile=10000,
           threads=None,
           cache_path=None,
           remove=True,
           encoding='utf-8'):
    """
    对target_path下的所有文件进行排序, 生成新的同名文件替换原来的文件，之后的文件去掉排序标号
    lines_per_tempfile: 排序时生成的每个临时文件中包含的文件行数
    cache_path: 排序时生成的临时文件的存放地址
    """
    if target_path is None:
      target_path = self.subfile_path
    if cache_path is None:
      cache_path = self.subfile_path
    if threads is None:
      threads = cpu_count()
    self.remove = remove
    start_time = time.time()
    print("Start Sorting Subfiles: This line only shows when 'sort' in class PreprocessVector is called.")
    p = pool.Pool(threads)
    # p.starmap_async(self._sort,
    # [(file, target_path, lines_per_tempfile, cache_path, encoding) for file in self.valid + '+'])
    for file in self.valid + '+':
      p.apply(self._sort, args=(file, target_path, lines_per_tempfile, cache_path, encoding))
    p.close()
    print("Sort Operation is Completed. Cost Time is {:.2f}s".format(time.time() - start_time))
    # if remove:
    # self._remove_cache(cache_path)

  def _remove_cache(self, cache_path):
    # 由于并行运算可能存在并行进程晚于主函数执行的情况，这里等待一段时间后再执行删除
    count_time = time.time()
    sub_cache_path = 'sub_cache_path'
    tmp_count = time.time()
    while time.time() - tmp_count < 5:
      continue
    while os.listdir(cache_path + '/' + sub_cache_path):
      tmp_count = time.time()
      while time.time() - tmp_count < 3:
        continue
    shutil.rmtree(cache_path + '/' + sub_cache_path)
    print("DELETE the cache data, this causes {:.2f}s "
          "delays because of waiting for the end of pool operation.".format(time.time() - count_time))

  def _sort(self, file, path, lines_per_tempfile, cache_path, encoding='utf-8'):
    sub_cache_path = 'sub_cache_path'  # 不要更改
    os.makedirs(cache_path + '/' + sub_cache_path, exist_ok=True)
    len_end_label = len(self.end_label)
    with open(path + '/' + file, 'r', encoding=encoding) as f:
      temp = [f.readline() for x in range(lines_per_tempfile)]
      lines = [line for line in temp if line]
      obj = {}
      count = 0
      while len(lines) > 0:
        # 以lines的每一行的第一个词为key，后续为value，生成字典
        dict = {}
        for line in lines:
          line = line.strip()
          if len(line) > 2 * len_end_label:
            # print('0 ', line[len_end_label:-len_end_label])
            if line[:len_end_label] == self.end_label and line[-len_end_label:] == self.end_label:
              line = line[len_end_label:-len_end_label].split(self.split_label)
              # print('1 ', line)
            else:
              continue
          else:
            continue
          # print('2 ', line)
          dict[int(line[0])] = line[1]
        lines = [str(key) + self.split_label + dict[key] for key in sorted(dict.keys())]
        obj[count] = open(cache_path + '/' + sub_cache_path + '/' + file + '_' + str(count), 'w', encoding=encoding)
        for line in lines:
          obj[count].write(line + '\n')
        obj[count].close()
        temp = [f.readline() for x in range(lines_per_tempfile)]
        lines = [line for line in temp if line]
        count += 1
    f.close()
    # 重新打开文件，进行读取和写入
    for count in obj.keys():
      obj[count] = open(cache_path + '/' + sub_cache_path + '/' + file + '_' + str(count), 'r', encoding=encoding)
    f = open(path + '/' + file, 'w', encoding=encoding)
    # 分别读取各个子文件中的line，每次读取一行，然后比较得到最小的一行，去除index写到f中
    # 第一次读取
    max_label = 1e10
    lines = [obj[count].readline().split(self.split_label) for count in range(len(obj.keys()))]
    line_dict = {count: int(lines[count][0]) for count in range(len(lines))}
    min_count = min(line_dict, key=line_dict.get)
    while self._stop(line_dict, max_label) is False:
      f.write(lines[min_count][1])
      lines[min_count] = obj[min_count].readline().split(self.split_label)
      # print(lines[min_count])
      line_dict[min_count] = int(lines[min_count][0]) if lines[min_count][0] else max_label
      min_count = min(line_dict, key=line_dict.get)
    f.close()
    for count in obj.keys():
      obj[count].close()
    if self.remove:
      for count in obj.keys():
        os.remove(cache_path + '/' + sub_cache_path + '/' + file + '_' + str(count))

  def _stop(self, line_dict, max_label):
    for value in line_dict.values():
      if value != max_label:
        return False
    return True

