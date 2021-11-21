"""
generate random walk paths based on the word co-occurrence network.

author: Dongyang Yan
github: github.com/fiyen
last modification date: 2021/11/04
"""

import numpy as np
import networkx as nx
import random
import time
import tensorflow as tf
from multiprocessing import pool, cpu_count
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Random Walk. Generate the random walk paths.
class RandomWalk:
  """
  get the random walk paths based on a network
  """
  def __init__(self, walks=10, length=10, extra_walks=1000, names=None):
    self.__walks = walks
    self.__length = length
    self.__names = names
    self.__extra_walks = extra_walks
    self.__extrowalked = 0
    self.neighbor_dis = {}  # 每个节点的邻居权重分布

  def trans_net(self, G):
    """
    transform G to networkx form
    :param G: matrix or networkx form
    :return: networkx form G
    """
    if isinstance(G, nx.classes.graph.Graph):
      return G
    elif G.shape[0] == G.shape[1]:
      if self.__names is not None:
        if len(self.__names) != G.shape[0]:
          raise ValueError("Adjacency matrix should contain number of nodes equals to names.")
        else:
          graph = nx.Graph()
          for i in range(G.shape[0]):
            for j in range(G.shape[1]):
              graph.add_edge(self.__names[i], self.__names[j])
      return graph

  def rand_paths(self, G, v, loop=False, weighted=False, smooth=False, reverse=False):
    """
    get random paths start from v
    :param G: network
    :param v: start node
    :param loop: whether allow loop in paths, True or False
    :param weighted: whether the random walk depends on the weight of the node, the high weighted node will be more
    likely chosen as next node while random walking.
    :param reverse: if weighted==True, the reverse option, Note: if reverse==False, the loop will be fitted to True.
    :return:
    """
    if reverse:
      loop = True
    paths = []
    if not loop:
      walks = 0
      extro_walks = 0
      while walks < self.__walks:
        path = [v]
        length = 1
        exists = [v]
        v_ = v
        while length < self.__length:
          neighbors = [x for x in nx.neighbors(G, v_)]
          if len(neighbors) > 0:
            dif = set(neighbors).difference(exists)
            if len(dif) > 0:
              v_ = random.sample(dif, 1)[0]
              path.append(v_)
              exists.append(v_)
              length += 1
            else:
              break
          else:
            break
        if length == self.__length:
          paths.append(path)
          walks += 1
        else:
          extro_walks += 1
        if extro_walks > self.__extra_walks:
          self.__extrowalked = extro_walks
          # 如果只选取一个随机游走路径，返回这个不全的随机游走路径
          if self.__walks == 1:
            paths.append(path)
            return paths
          break
    else:
      walks = 0
      extro_walks = 0
      while walks < self.__walks:
        path = [v]
        length = 1
        v_ = v
        while length < self.__length:
          neighbors = [x for x in nx.neighbors(G, v_)]
          if len(neighbors) > 0:
            if weighted:
                v_ = self.weighted_choice(G, v, reverse=reverse, smooth=smooth, neighbors=neighbors)
            else:
                v_ = random.sample(neighbors, 1)[0]
            path.append(v_)
            length += 1
          else:
            break
        if length == self.__length:
          paths.append(path)
          walks += 1
        else:
          extro_walks += 1
        if extro_walks > self.__extra_walks:
          self.__extrowalked = extro_walks
          # 如果只选取一个随机游走路径，返回这个不全的随机游走路径
          if self.__walks == 1:
            paths.append(path)
            return paths
          break
    return paths

  def all_walks(self, G, loop=False):
    allwalks = []
    for node in nx.nodes(G):
      for walk in self.rand_paths(G, node, loop):
        allwalks.append(walk)
    return allwalks

  def weighted_choice(self, G, node, reverse=False, smooth=False, neighbors=None, alpha=4/3):
    """
    Choose the next node to random walk according to the weight of nodes.
    :param alpha:
    :param smooth: elevate the influence of nodes with too high frequency
    :param neighbors: To skip to acquire neighbors if neighbors is provided.
    :param G:
    :param nodes:
    :param reverse:
    :return:
    """
    if node in self.neighbor_dis.keys():
      weight_dis = self.neighbor_dis[node]
    else:
      if neighbors is not None:
        nodes = neighbors
      else:
        nodes = [n for n in G.neighbors(node)]
      if not reverse:
        weight = [(node, (G.degree[node])**alpha) for node in nodes]
      else:
        weight = [(node, (1 / (G.degree[node]+1))**alpha) for node in nodes]
      total = sum([w for _, w in weight])
      weight_dis = [(word, w / total) for word, w in weight]
      if not smooth:
        tem = 0
        for index, wd in enumerate(weight_dis):
          tem += wd[1]
          weight_dis[index] = (wd[0], tem)
      self.neighbor_dis[node] = weight_dis
    if not smooth:
      seed = np.random.rand()
      for node, w in weight_dis:
        if seed < w:
          return node
    else:
      seed = np.random.rand()
      sample = random.sample(weight_dis, 1)[0]
      if seed < sample[1]:
        return sample[0]
      else:
        while True:
          sample_ = random.sample(weight_dis, 1)[0]
          if sample_[0] != sample[0]:
            return sample[0]


# Node2Vec Walk. Generate pathes with Node2Vec method.
def alias_setup(probs):
  """
  :param probs: list
  :return:
  """
  K = len(probs)
  q = np.zeros(K)
  J = np.zeros(K, dtype=np.int)

  smaller = []
  larger = []
  for kk, prob in enumerate(probs):
    q[kk] = K * prob
    if q[kk] < 1.0:
      smaller.append(kk)
    else:
      larger.append(kk)

  while len(smaller) > 0 and len(larger) > 0:
    small = smaller.pop()
    large = larger.pop()

    J[small] = large
    q[large] = q[large] + q[small] - 1.0
    if q[large] < 1.0:
      smaller.append(large)
    else:
      larger.append(large)
  return J, q


def alias_draw(J, q):
  """
  Draw sample from a non-uniform discrete distribution using alias sampling.
  :param J:
  :param q:
  :return:
  """
  K = len(J)
  kk = int(np.floor(np.random.rand() * K))
  if np.random.rand() < q[kk]:
    return kk
  else:
    return J[kk]


class AliasSamplingForNet:
  """
  A method to quickly sample from non-uniform discrete distribution
  Specially designed for net
  :return:
  """

  def __init__(self, p, q, weight_reverse=False, name=None):
    self.p = p
    self.q = q
    self.weight_reverse = weight_reverse
    self.name = name
    self.fitted = False

  def fit_net(self, net):
    """
    net is a networkx.Graph or networkx.DiGraph
    :param net:
    :return:
    """
    if isinstance(net, nx.Graph):
      directed = False
    elif isinstance(net, nx.DiGraph):
      directed = True
    else:
      raise TypeError('Only support input type "networkx.Graph" or "networkx.DiGraph". '
                      'But the input type is {}'.format(type(net)))
    self._2nd_option = {}  # if node is from the start, use this to sample the node
    for node in net.nodes:
      if not self.weight_reverse:
        edge_weight = [net[node][neig]['weight'] for neig in sorted(net.neighbors(node))]
      else:
        edge_weight = [net[node][neig]['weight'] for neig in sorted(net.neighbors(node))]
        ave = np.mean(edge_weight)
        edge_weight = [v - (ave - v) for v in edge_weight]
      neig_probs = self.normalize(edge_weight)
      self._2nd_option[node] = alias_setup(neig_probs)
    self._3rd_option = {}  # if node is after the 2rd node along the path, use this to sample the node.
    for src, _2nd in net.edges:
      if directed:
        self._3rd_option[(src, _2nd)] = alias_setup(self._get_edge_probs(net, src, _2nd))
      else:
        self._3rd_option[(src, _2nd)] = alias_setup(self._get_edge_probs(net, src, _2nd))
        self._3rd_option[(_2nd, src)] = alias_setup(self._get_edge_probs(net, _2nd, src))
    self.fitted = True

  def _get_edge_probs(self, net, src, _2nd):
    """
    get probs of an edge
    :param net:
    :param src:
    :param _2nd:
    :return:
    """
    ngrs = sorted(net.neighbors(_2nd))
    if not self.weight_reverse:
      edge_weight = [net[_2nd][neig]['weight'] for neig in ngrs]
    else:
      edge_weight = [net[_2nd][neig]['weight'] for neig in ngrs]
      ave = np.mean(edge_weight)
      edge_weight = [v - (ave - v) for v in edge_weight]
    neig_probs = []
    for ind, n in enumerate(ngrs):
      if n == src:
        neig_probs.append(edge_weight[ind] / self.p)
      elif net.has_edge(_2nd, n):
        neig_probs.append(edge_weight[ind])
      else:
        neig_probs.append(edge_weight[ind] / self.q)
    neig_probs = self.normalize(neig_probs)
    return neig_probs

  def normalize(self, x):
    """
    normalize x
    :param x:
    :return:
    """
    return np.array(x) / np.sum(x)

  def sample(self, net, v1, v2=None):
    """
    sample the next node for a path
    :param net: should be networkx.Graph or networkx.DiGraph
    :param v1:
    :param v2:
    :return:
    """
    if isinstance(net, nx.Graph) or isinstance(net, nx.DiGraph):
      pass
    else:
      raise TypeError('Only support input type "networkx.Graph" or "networkx.DiGraph". '
                      'But the input type is {}'.format(type(net)))
    if not self.fitted:
      self.fit_net(net)
    if v2 is None:
      ngrs = sorted(net.neighbors(v1))
      return ngrs[alias_draw(self._2nd_option[v1][0], self._2nd_option[v1][1])]
    else:
      ngrs = sorted(net.neighbors(v2))
      return ngrs[alias_draw(self._3rd_option[(v1, v2)][0], self._3rd_option[(v1, v2)][1])]


class Node2VecWalk(AliasSamplingForNet):
  """
  To generate a series of paths with node2vec method.
  p, q: hyper-parameters that control the walk way, large p will cause the walk less likely to step back while large q
  will cause the walk step nearby the start node
  weight_reverse: whether reverse the weight, which means that larger edge weight will less likely to be visited.
  names: if the form of net is adjacent matrix, the names will be the label of all the nodes, the index is corresponded
  to the index in the matrix
  """

  def __init__(self, p, q, walks=10, length=10, weight_reverse=False, names=None):
    super(Node2VecWalk, self).__init__(p=p, q=q, weight_reverse=weight_reverse)
    self.p = p
    self.q = q
    self.weight_reverse = weight_reverse
    self.__walks = walks
    self.__length = length
    self.__names = names
    self.__extrowalked = 0

  def trans_net(self, G):
    """
    transform G to networkx form
    :param G: matrix or networkx form
    :return: networkx form G
    """
    if isinstance(G, nx.classes.graph.Graph):
      return G
    elif G.shape[0] == G.shape[1]:
      if self.__names is not None:
        if len(self.__names) != G.shape[0]:
          raise ValueError("Adjacency matrix should contain number of nodes equals to names.")
        else:
          graph = nx.Graph()
          for i in range(G.shape[0]):
            for j in range(G.shape[1]):
              graph.add_edge(self.__names[i], self.__names[j])
        return graph

  def rand_paths(self, G, v):
    """
    get random paths start from v
    :param G: network
    :param v: start node
    :return:
    """
    path = [v]
    length = 0
    while length < self.__length:
      if length < 1:
        v_ = self.sample(G, path[-1])
      else:
        v_ = self.sample(G, path[-2], path[-1])
      path.append(v_)
      length += 1
    return path

  def gen_walks(self, G, batch_size=1000):
    num = 0
    allwalks = []
    G = self.trans_net(G)
    node = [n for n in nx.nodes(G)]
    for i in range(self.__walks):
      random.shuffle(node)
      for n in node:
        allwalks.append(self.rand_paths(G, n))
        num += 1
        if num % batch_size == 0:
          yield allwalks
          allwalks = []
          num = 0
    if len(allwalks) > 0:
      yield allwalks


# PathGen. Generate random walk pathes as the inputs of pcnn net model.
class PathGen:
  """
  产生随机游走路径，作为数据预处理，输出的结果作为pcnn_net_model的输入。建立网络使用的是动态词共现网络。
  / To generate random paths of data preprocessing. The output is to be the input of pcnn_net_model. The newtork is word
  co-occurrence complex network.
  number: 每一个文本网络随机游走路径的生成数量 / The number of random walk paths for each sample.
  length: 随机游走路径的长度 / length of random walk paths
  walk: 随机游走方式：'normal', 'self_avoiding', 'weighted', 'weighted_reverse','smooth', 'smooth_reverse','node2vec',默认为self_avoiding
    / The choice method for every step during the random walk. 'normal', 'self_avoiding', 'weighted',
    'weighted_reverse','smooth', 'smooth_reverse','node2vec',Default: self_avoiding
  p,q: 当walk='node2vec'时需要填写的项，控制node2vec游走方式的超参数，如果p大，则游走趋向于不往回走，q大则使游走在起点周围进行。
    Must be set when walk='node2vec'. The parameter to control the stepping trend for every step of random walk.
    If p is larger, the trend is avoid to go back; if q is larger, the trend is to walk around the start node.
  start_nodes: 随机游走的开始节点, 格式（batches，number）注意：如果start_nodes不为None，则无法使用data_expansion.
    / Start node of the random walk. Format: (batches, number). Note: If start_nodes is not None, the data_expansion
    is ignored.
  expansion_ratio: 数据倍增比例，=n即将数据倍增为n倍, <=1则不使用倍增 / The ratio to expansion data, just simply copy ratio times.
    if =n > 1, to copy the data n times; if <= 1, do not copy.
  sample_cross: 是否用两个同标签样本的拼接形成的网络模型进行倍增取样,注意：sample_cross只有在adaptive_expansion开启时才有效 / Whether
    to use two samples in the same class to construct synthetic complex network and then to generate random walk paths.
    Note: sample_cross only be actived if adaptive_expansion is activated.
  adaptive_expansion:  是否开启自适应数据倍增，自适应数据倍增是将数据集中各类别数据的数量倍增到基本一致。默认为False，如果
    为True，则数据会先按类整理，然后以数据量最多的类别为基准Cb，将其他数据的数量增至与该数据的数据量相同。其思路如下：
    首先，如果Cb的数据量是Ci的n倍多m,则先将Ci的数据量倍增n倍，然后在Ci的源数据中取样m个。
    注意：当adaptive_expansion为True时，expansion_ratio需要<=1. 如果expansion_ratio=n>1, 则会在adaptive_expansion操作结束后，
    对数据再进行n倍的倍增
    \ Whether to adopt the strategy to expand adaptive number of samples for each class to make the number of samples in
    each class approximately equal. Default: False. If adaptive_expansion=True, the data will be arranged to groups, in
    which samples are with the same class. The samples of each group is expanded to the number equal to the group with
    the most samples.
    Note: the effect of adaptive_expansion and expansion_ratio can be superimposed. So if adaptive_expansion is True, make
    sure that expansion_ratio <= 1. Otherwise, after adaptive_expansion operation is completed, the data is copied n
    times again.
  cut_off_ratio: 截断倍率，当adaptive_expansion=True时有效，为了避免样本数量极少的类别出现过多重复数据，对小样本倍增的倍率
    进行限制，n超过cut_off_ratio后，只倍增cut_off_ratio倍。默认为0，表示不做限制。当cut_off_ratio大于0的时候，该选项才会起作用
    同时，规定cut_off_ratio为整数
    \ (int) take effect if adaptive_expansion is True to make the samples' number of minority class to be expanded with
    expanding times cutting off by a max value. Default: 0, do not do cut off operation.
  cut_off_num: 截断数量，和cut_off_ratio作用相同，不同点在于，cut_off_num从数量上限制倍增，如果cut_off_num>0,则所有的倍增
    样本的数量都只增加到cut_off_num
    \ (int) take effect if adaptive_expansion is True to make the samples' number of minority class cut off by a max
    value. Default: 0, do not do cut off operation.
    注意：cut_off_num和cut_off_ratio是可以同时起作用的 / cut_off_num and cut_off_ratio can be activated simultaneously.
    注意：如果训练数据有labels，labels倍增可以用self.multiple_objs操作，操作后的数据与倍增后的训练数据一一对应 / If the training data
    contains labels, the labels can be expanded with self.multiple_objs, with the label mapping to the training samples
    one-by-one.
  fast_mode: 是否开启并行模式 / whether to take multiprocess (for cpu)
  threads: 并行线程数，如果为0，表示开启cpu的最大线程数量。默认为0 / threads. If =0, take the max thread number of cpu
  try_times: 有些self_avoiding的路径无法达到length的长度，需要尝试try_times次，如果try_times过后仍然未达到length的长度，则
    将未达到长度的节点padding / for self_avoiding walk. Some random walk can not reach length, which has to be handled by
    try many times. If length is still not reached after try_times trial, padding the last length.
  except_padding: 当try_times次过后仍未达到length的长度，其余长度填充的方式，‘random’，随机选择节点填充，‘zeros’，填充0 /
    If length is still not reached after try_times trial, padding the last length with this value.
  undersampling: 是否对个数大于cut_off_num的类进行下采样，使其数量减少至cut_off_num, 仅当cut_off_num>0, adaptive_expansion==True
    时有效 / whether to undersample for the class with the samples number larger than  cut_off_num (to decease the value to
    cut_off_num). Activated only if cut_off_num > 0 and adaptive_expansion==True.
  """
  def __init__(self, number,
            length,
            walk='self_avoiding',
            start_nodes=None,
            expansion_ratio=1,
            adaptive_expansion=False,
            sample_cross=False,
            cut_off_ratio=0,
            cut_off_num=0,
            threads=0,
            try_times=10,
            except_padding='zeros',
            undersampling=False,
            p=1,
            q=1,
            ):
    self.number = number
    self.length = length
    self.walk = walk
    self.start_nodes = start_nodes
    self.ratio = expansion_ratio
    self.adaptive = adaptive_expansion
    self.sample_cross = sample_cross
    self.cut_ratio = cut_off_ratio
    self.cut_num = cut_off_num
    self.threads = threads
    self.try_times = try_times
    self.except_padding = except_padding
    self.undersampling = undersampling
    self.p = p
    self.q = q
    if threads == 1 or threads < 0:
      self.fast_mode = False
    else:
      self.fast_mode = True

    if self.start_nodes is not None:
      for index, nodes in enumerate(self.start_nodes):
        if isinstance(nodes, list):
          len_ = len(nodes)
        elif isinstance(nodes, np.ndarray):
          len_ = nodes.shape[0]
        else:
          raise TypeError('Please use list or ndarray instead.')
        if len_ != self.number:
          raise ValueError('The number of start nodes in line {} is not equal to wanted number of paths.'
                              .format(index))
    if self.except_padding not in ['random', 'zeros']:
      raise ValueError('Only support {random, zeros} padding. Error padding: ', self.except_padding)
    if self.ratio > 1:
      if type(self.ratio) is not int:
        raise ValueError('Type of expansion_ratio should be int but is ', type(self.ratio))
      if self.start_nodes is not None:
        print('\033[1;31m WARNING: When calling expansion operation, the start nodes is randomly given. '
              'So the input value of start_nodes is neglected! \033[0m')
    if isinstance(self.cut_ratio, float):
      self.cut_ratio = int(self.cut_ratio)
      print('\033[1;31m WARNING: The value of cut_off_ratio is float. It is transferred to int. \033[0m')
    if isinstance(self.cut_num, float):
      self.cut_num = int(self.cut_num)
      print('\033[1;31m WARNING: The value of cut_off_ratio is float. It is transferred to int. \033[0m')
    #print('padding: ',except_padding)

  def __call__(self, inputs, labels=None, verbose=False, batch_size=256, **kwargs):
    """
    :param inputs: (batches, length), list或np.array格式
    :param kwargs:
    :return: self.number random paths with self.length.
    """
    if verbose:
      print('Calling PathGen......')
    t = time.time()
    inputs = tf.constant(inputs)
    if self.threads <= 0:
      threads = cpu_count() - 1
    else:
      threads = self.threads
    nets = [self._trans_net(line) for line in inputs]
    if self.start_nodes is not None:
      nets = np.array(nets)
    else:
      if self.adaptive is False:
        nets = self.multiple_obj(nets)
        nets = np.array(nets)
        if labels is not None:
          labels = self.multiple_obj(labels)
          labels = np.array(labels)
      else:
        if labels is None:
          raise ValueError("labels must be input if adaptive_expansion is True.")
        else:
          if np.ndim(labels) <= 1:
            labels = self.one_hot_labeling(labels)
          sum_classes = np.sum(labels, axis=0)
          num_classes = sum_classes.shape[0]
          max_class = np.argmax(sum_classes)
          max_num = int(np.max(sum_classes))
          sort_data = {i: {'data': [], 'labels': []} for i in range(num_classes)}
          for index, label in enumerate(labels):
            i = np.argmax(label)
            sort_data[i]['data'].append(nets[index])
            sort_data[i]['labels'].append(label)
          if self.cut_ratio <= 0:
            cut_ratio = 999999999999
          else:
            cut_ratio = self.cut_ratio
          for i, num in enumerate(sum_classes):
            num = int(num)
            r = max_num % num
            n = max_num // num
            if self.cut_num > 0:
              if self.cut_num < num:
                r = 0
                n = 1
              else:
                r = self.cut_num % num
                n = self.cut_num // num
            if not self.sample_cross:
              if n <= cut_ratio:
                indexes = [x for x in range(num)]
                sample_indexes = random.sample(indexes, r)
                r_data = [sort_data[i]['data'][x] for x in sample_indexes]
                r_labels = [sort_data[i]['labels'][x] for x in sample_indexes]
                sort_data[i]['data'] = self.multiple_obj(sort_data[i]['data'], n) + r_data
                sort_data[i]['labels'] = self.multiple_obj(sort_data[i]['labels'], n) + r_labels
              else:
                sort_data[i]['data'] = self.multiple_obj(sort_data[i]['data'], cut_ratio)
                sort_data[i]['labels'] = self.multiple_obj(sort_data[i]['labels'], cut_ratio)
            else:
              if n <= cut_ratio:
                num_sample = (n - 1) * num + r
              else:
                num_sample = (cut_ratio - 1) * num

              indexes = [x for x in range(num)]
              merge_data = []
              merge_labels = []
              for count in range(num_sample):
                if len(indexes) > 1:
                  sample_indexes = random.sample(indexes, 2)
                else:
                  sample_indexes = [0, 0]
                merge_data.append(self.net_merge([sort_data[i]['data'][x] for x in sample_indexes]))
                merge_labels.append(sort_data[i]['labels'][0])
              sort_data[i]['data'] = sort_data[i]['data'] + merge_data
              sort_data[i]['labels'] = sort_data[i]['labels'] + merge_labels
          if self.cut_num > 0 and self.adaptive and self.undersampling:
            for i in sort_data.keys():
              if len(sort_data[i]['data']) > self.cut_num:
                us_data = random.sample(sort_data[i]['data'], self.cut_num)
                sort_data[i]['data'] = us_data
                sort_data[i]['labels'] = sort_data[i]['labels'][:self.cut_num]
          new_nets = []
          new_labels = []
          for i in range(num_classes):
            new_nets += sort_data[i]['data']
            new_labels += sort_data[i]['labels']
          # shuffles
          new_nets = np.array(new_nets)
          new_labels = np.array(new_labels)
          new_nets = self.multiple_obj(new_nets)
          new_nets = np.array(new_nets)
          new_labels = self.multiple_obj(new_labels)
          new_labels = np.array(new_labels)
          index_new = [i for i in range(new_nets.shape[0])]
          random.shuffle(index_new)
          nets = np.array([new_nets[i] for i in index_new])
          labels = np.array([new_labels[i] for i in index_new])
    nets_size = nets.shape[0]
    batch_num = int(np.ceil(nets_size / batch_size))
    cumulate_len = 0
    cumulate_time = 0
    out = []
    p = pool.Pool(threads)
    for i in range(batch_num):
      start = time.time()
      net_batch = nets[i*batch_size:(i+1)*batch_size]
      len_ = net_batch.shape[0]
      if self.start_nodes is None:
        starts_batch = [None for i in range(len_)]
      else:
        starts_batch = self.start_nodes[i*batch_size:(i+1)*batch_size]
      if not self.fast_mode:
        out = out + [self._pool_for_call(net, np.array(start_nodes))
                    for net, start_nodes in zip(net_batch, starts_batch)]
      else:
        out_ = p.map(self._wrap_for_pool, zip(net_batch, starts_batch))
        out = out + out_
      cumulate_len = cumulate_len + len_
      if verbose:
        if i < batch_num - 1:
          cumulate_time = cumulate_time + time.time() - start
          print('{}/{} - ETA: {:.0f}s.'.
              format(str(cumulate_len).rjust(len(str(nets_size))), nets_size,
                    (nets_size - cumulate_len) * cumulate_time / cumulate_len,))
        else:
          cumulate_time = cumulate_time + time.time() - start
          print('{}/{} - {:.0f}s.'.
              format(str(cumulate_len).rjust(len(str(nets_size))), nets_size,
                    cumulate_time))
    p.close()
    p.join()
    out = np.array(out)
    if verbose:
      print("PathGen finished. Time costs: {:.2f}s.".format(time.time() - t))
    return out, labels

  def one_hot_labeling(self, y):
    all_label = dict()
    for label in y:
      if label not in all_label:
        all_label[label] = len(all_label) + 1
    one_hot = np.identity(len(all_label))
    y = [one_hot[all_label[label] - 1] for label in y]
    return y

  def multiple_obj(self, objs, n=0):
    """
    objs中的元素倍增后返回
    :param n:
    :param objs: np.array, list
    :return:
    """
    if n <= 0:
      n = self.ratio
    if not isinstance(objs, (list, np.ndarray)):
      raise AttributeError('Only support {list, ndarray} type. objs" type is ', type(objs))
    if not isinstance(objs, list):
      objs = objs.tolist()
    out = []
    for i in range(n):
      out = out + objs
    return out

  def net_merge(self, Gs):
    """
    将几个网络G1和G2合并成一个网络，即整合两个网络中的所有点以及边
    Gs: 存有若干网络模型的列表
    """
    edges = []
    for g in Gs:
      edges += [e for e in g.edges]
    edges = [e + (1, ) for e in edges]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G

  def _pool_for_call(self, net, start_nodes):
    """
    并行操作函数
    :param line:
    :param start: 起始点
    :return:
    """
    #net = self._trans_net(line)
    start_nodes = np.array(start_nodes)
    get_path = RandomWalk(walks=1, length=self.length, extra_walks=self.try_times)
    if not start_nodes:
      start_nodes = self._choose_start_nodes(net)
    if self.walk == 'self_avoiding':
      loop = False
      weighted = False
      reverse = False
      smooth = False
    elif self.walk == 'normal':
      loop = True
      weighted = False
      reverse = False
      smooth = False
    elif self.walk == 'weighted':
      loop = True
      weighted = True
      reverse = False
      smooth = False
    elif self.walk == 'weighted_reverse':
      loop = True
      weighted = True
      reverse = True
      smooth = False
    elif self.walk == 'smooth':
      loop = True
      weighted = True
      reverse = False
      smooth = True
    elif self.walk == 'smooth_reverse':
      loop = True
      weighted = True
      reverse = True
      smooth = True
    elif self.walk == 'node2vec':
      get_path = Node2VecWalk(walks=1, length=self.length, p=self.p, q=self.q)
    else:
      raise ValueError('Not support {} walk'.format(self.walk))
    if self.walk == 'node2vec':
      paths = [get_path.rand_paths(net, s) for s in start_nodes]
    else:
      paths = [get_path.rand_paths(net, s, loop=loop, weighted=weighted, smooth=smooth, reverse=reverse)[0] for s in start_nodes]
    # 纠错
    for index, path in enumerate(paths):
      if len(path) < self.length:
        while len(path) < self.length:
          if self.except_padding == 'random':
            path.append(random.sample(net.nodes, 1)[0])
          else:
            path.append(0)
        paths[index] = path
    return paths

  def _wrap_for_pool(self, args):
    return self._pool_for_call(*args)

  def _trans_net(self, line):
    line = np.array(line)
    net = nx.Graph()
    for index, node in enumerate(line):
      if index > 0:
        try:
          #net.add_edge(line[index-1], node)
          net.add_weighted_edges_from([(line[index-1], node, 1)])
        except TypeError:
          print('line', line)
    return net

  def _choose_start_nodes(self, net):
    """
    得到line生成随机游走序列的左右开始节点
    :param net:
    :return:
    """
    nodes = [x for x in net.nodes]
    if 0 in nodes:
      nodes.remove(0)
    return [random.sample(nodes, 1)[0] for i in range(self.number)]