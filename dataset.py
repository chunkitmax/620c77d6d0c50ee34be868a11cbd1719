'''
dataset.py
read data from files
'''

import os.path as Path
import re
import sys
from collections import Counter
from zipfile import ZipFile

import torch.utils.data as Data

from log import Logger


class Dataset(Data.Dataset):
  def __init__(self, raw_data, do_cleaning=True):
    # zipf = ZipFile('hw3_dataset.zip', 'r')
    # train_data = zipf.read('train.csv')
    # test_data = zipf.read('test_for_you_guys.csv')
    self.label = []
    self.data = []

    sentiments = set()
    self.word_counter = Counter()

    Log = Logger()
    Log.i('Start loading dataset...')

    # load lists if saved previously
    has_lists = Path.exists('sentiment_list') and Path.exists('word_list')
    if has_lists:
      Log.d('Sentiment and word list found!')
      with open('sentiment_list', 'r') as sf:
        self.sentiments = sf.read().strip().split('\n')
      tmp_dict = {}
      with open('word_list', 'r') as wf:
        for line in wf.readlines():
          word, freq = line.strip().split()
          tmp_dict[word] = int(freq)
        self.word_counter = Counter(tmp_dict)
        self.word_list = ['<unk>'] + \
                          [key for key, value in self.word_counter.items() if value >= 3]
        del tmp_dict
      if len(self.sentiments) == 0 or len(self.word_list) == 0:
        raise AssertionError('either sentiment and word list is empty') \
              .with_traceback(sys.exc_info()[2])
      self.sentiments = {word: index for index, word in enumerate(self.sentiments)}
      self.word_list = {word: index for index, word in enumerate(self.word_list)}

    raw_data = raw_data.decode('utf-8').strip().split('\n')[1:]
    data_len = len(raw_data)
    # add data and label to array
    for index, line in enumerate(raw_data):
      cols = line.split(',', 3)
      if do_cleaning:
        words = self._clean_str(cols[3].strip('"')).split()
      else:
        words = cols[3].strip('"').split()
      # tweet_id and authour ignore?
      if not has_lists:
        sentiments.add(cols[1])
        self.label.append([cols[1]])
        self.word_counter += Counter(words)
        self.data.append(words)
      else:
        self.label.append([self.sentiments[cols[1]]])
        self.data.append([self.word_list[word] if word in self.word_list else 0
                          for word in words])
      Log.i('Loading %6d / %6d'%(index, data_len))
    Log.i('Finish loading', True)

    Log.i('Start preprocessing...')

    if not has_lists:
      # denoise by setting minimum freq
      self.word_list = ['<unk>'] + [key for key, value in self.word_counter.items() if value >= 3]

      # save sentiment and word list
      self.sentiments = list(sentiments)
      if len(self.sentiments) > 0 and len(self.word_list) > 0:
        with open('sentiment_list', 'w+') as sf:
          for sentiment in self.sentiments:
            sf.write(sentiment+'\n')
        with open('word_list', 'w+') as wf:
          for word, freq in dict(self.word_counter).items():
            wf.write(word+' '+str(freq)+'\n')
      else:
        raise AssertionError('either sentiment and word list is empty') \
              .with_traceback(sys.exc_info()[2])

      # convert to dict for fast searching
      self.sentiments = {word: index for index, word in enumerate(self.sentiments)}
      self.word_list = {word: index for index, word in enumerate(self.word_list)}

      # convert text to index
      for index, [data_ent, label_ent] in enumerate(zip(self.data, self.label)):
        # <unk> (index 0) if word not found
        self.data[index] = [self.word_list[word] if word in self.word_list else 0
                            for word in data_ent]
        self.label[index] = [self.sentiments[word] for word in label_ent]

    Log.i('Finish preprocessing', True)
  def __getitem__(self, index):
    return self.data[index], self.label[index]

  def __len__(self):
    return len(self.data)

  def _clean_str(self, string):
    '''
    Remove noise from input string
    '''
    string = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}'
                    +r'\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', '<url>', string)
    string = re.sub(r'&[a-zA-Z];', ' ', string)
    string = re.sub(r'[^A-Za-z0-9,!?\(\)\.\'\`]', ' ', string)
    string = re.sub(r'[0-9]+', ' <num> ', string)
    string = re.sub(r'( \' ?)|( ?\' )', ' ', string)
    string = re.sub(r'(\'s|\'ve|n\'t|\'re|\'d|\'ll|\.|,|!|\?|\(|\))',
                    r' \1 ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    string = re.sub(r'(\. ?){2,}', '...', string)
    return string.strip().lower()

if __name__ == '__main__':
  pass
