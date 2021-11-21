"""
process the emotions datasets and form a new datasets.
"""
from nltk.stem import WordNetLemmatizer, PorterStemmer
lemmatool = WordNetLemmatizer()
stemmer = PorterStemmer()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stopwords = [w for w in stopwords.words('english')]


def lemmatize(text):
  """
  lammatize the text to its original form.
  :param text:
  :return:
  """
  text = text.split()
  text = [lemmatool.lemmatize(w.lower()) for w in text]
  text = ' '.join(text)
  return text


def remove_stopwords(text):
  """
  remove stopwords
  :param text:
  :return:
  """
  text = text.lower()
  texts = word_tokenize(text)
  texts = [w for w in texts if w not in stopwords]
  text = ' '.join(texts)
  return text


if __name__ == '__main__':
  emotions = ['anger', 'fear', 'joy', 'sadness']

  text_form = 'stemmed'  # no-stop, stemmed, all-terms

  # For train data
  file_dir = "D:/data/imbalancedData/datasets/emotions/"
  save_file_dir = 'D:/data/imbalancedData/datasets/'
  train_texts = []
  train_labels = []
  for emo in emotions:
    samples = open(file_dir + '%s.train.txt'%emo, 'r', encoding='utf8').readlines()
    for sample in samples:
      _, text, label, _ = sample.strip().split('\t')
      if text_form == 'no-stop' or text_form == 'stemmed':
        text = remove_stopwords(text)
        if text_form == 'stemmed':
          text = lemmatize(text)
      train_texts.append(text)
      train_labels.append(label)
  with open(save_file_dir + 'emotions-train-%s.txt'%text_form, 'w', encoding='utf8') as f:
    for text, label in zip(train_texts, train_labels):
      f.write('%s\t%s\n'%(label, text))
    f.close()

  # For test data
  test_texts = []
  test_labels = []
  for emo in emotions:
    samples = open(file_dir + '%s.test.txt'%emo, 'r', encoding='utf').readlines()
    for sample in samples:
      _, text, label, _ = sample.strip().split('\t')
      if text_form == 'no-stop' or text_form == 'stemmed':
        text = remove_stopwords(text)
        if text_form == 'stemmed':
          text = lemmatize(text)
      test_texts.append(text)
      test_labels.append(label)
  with open(save_file_dir + 'emotions-test-%s.txt'%text_form, 'w', encoding='utf8') as f:
    for text, label in zip(train_texts, train_labels):
      f.write('%s\t%s\n'%(label, text))
    f.close()

