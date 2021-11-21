"""
random walk oversampling methods.
RWO-sampling: A random walk over-sampling approach to imbalanced data
classification. Zhang H., Li M., Information Fusion, 2014 (20) 99-116.

author: Dongyang Yan
github: github.com/fiyen
latest modification date: 2021/11/14
"""
import numpy as np


class RWO:
  """
  Random walk oversampling methods.
  """
  def data_expand(self, data, label, cut_num=0):
    """
    To expand the minority classes
    :param cut_num: the amount of every minority class will be expanded less than cut_num
    :param data: features vectors
    :param label: one hot vectors
    :return: np.array
    """
    class_num = len(list(label[0]))
    sorted_data = {x: {'data': [], 'label': []} for x in range(class_num)}
    for ind, d in enumerate(data):
      sorted_data[np.argmax(label[ind])]['data'].append(d)
      sorted_data[np.argmax(label[ind])]['label'] = label[ind]
    for i in range(class_num):
      sorted_data[i]['data'] = self.oversample(sorted_data[i]['data'], cut_num)
    new_data = []
    new_label = []
    for i in range(class_num):
      label = sorted_data[i]['label']
      for line in sorted_data[i]['data']:
        new_data.append(line)
        new_label.append(label)
    len_data = len(new_data)
    new_data = np.array(new_data)
    new_label = np.array(new_label)
    indices = np.random.permutation(range(len_data))
    new_data = new_data[indices]
    new_label = new_label[indices]
    return new_data, new_label

  def oversample(self, minority, cut_num):
    """
     increase the sample size to 'cut_num'
     :param cut_num:
     :param minority:
     :return:
     Note: if cut_num <= the size of minority, return minority itself directly
     """
    T = len(minority)
    if T <= 0:
      raise ValueError("The input minority has no samples!")
    if cut_num <= T:
      return minority
    else:
      samples = [v for v in minority]
      n = int(cut_num // T)
      r = int(cut_num % T)
      mean, stdvar = self.stats(samples)
      for i in range(n-1):
        for j in range(T):
          new_sample = [samples[j][d] - stdvar[d] / (T**0.5) * np.random.randn()
                        for d in range(len(samples[0]))]
          samples.append(new_sample)
      for _ in range(r):
        i = np.random.randint(0, T)
        new_sample = [samples[i][d] - stdvar[d] / (T ** 0.5) * np.random.randn()
                      for d in range(len(samples[0]))]
        samples.append(new_sample)
      return samples

  def stats(self, minority):
    """
    statistics of minority, mean and standard variance
    :param minority:
    :return:
    """
    if minority:
      dim = len(minority[0])
      mean = []
      stdvar = []
      for d in range(dim):
        dimv = [sample[d] for sample in minority]
        mean.append(sum(dimv) / len(dimv))
        stdvar.append((sum([(sample[d] - mean[d])**2 for sample in minority]) / len(dimv))**0.5)
      return mean, stdvar
    else:
      return None


if __name__ == "__main__":
  rwo = RWO()
  data = [[1,1,1,1,1,1], [2,2,2,2,2,2]]
  new_data = rwo.oversample(data, 30)
