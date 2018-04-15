'''
dataset.py
read data from files
'''

import os.path as Path
import re
import sys
from collections import Counter
from zipfile import ZipFile

import numpy as np
import torch as T
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

from log import Logger
from preprocess import _clean_str


class Dataset(Data.Dataset):
  def __init__(self, raw_data, train_valid_ratio=.2, do_cleaning=True, **args):
    if '_empty' not in args or not args['_empty']:
      self.label = []
      self.data = []
      self.valid_data = []
      self.valid_label = []

      sentiments = set()
      self.word_counter = Counter()
      self.max_len = 0

      Log = Logger()
      Log.i('Start loading dataset...')

      # Load lists if saved previously
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
          self.word_list = ['<pad>', '<unk>'] + \
                           [key for key, value in self.word_counter.items() if value >= 3]
          del tmp_dict
        if len(self.sentiments) == 0 or len(self.word_list) == 0:
          raise AssertionError('either sentiment and word list is empty') \
                .with_traceback(sys.exc_info()[2])
        self.sentiments = {word: index for index, word in enumerate(self.sentiments)}
        self.word_list = {word: index for index, word in enumerate(self.word_list)}

      if isinstance(raw_data, str):
        raw_data = raw_data.strip().split('\n')[1:]
      raw_data, valid_raw_data = train_test_split(raw_data, test_size=train_valid_ratio,
                                                  random_state=0)
      data_len = len(raw_data)
      valid_data_len = len(valid_raw_data)
      # Add data and label to array
      for index, line in enumerate(raw_data):
        cols = line.split(',', 3)
        if do_cleaning:
          words = _clean_str(cols[3].strip('"')).split()
        else:
          words = cols[3].strip('"').split()
        self.max_len = max(self.max_len, len(words))
        # Tweet_id and authour ignore?
        if not has_lists:
          sentiments.add(cols[1])
          self.label.append([cols[1]])
          self.word_counter += Counter(words)
          self.data.append(words)
        else:
          self.label.append([self.sentiments[cols[1]]])
          self.data.append([self.word_list[word] if word in self.word_list
                            else self.word_list['<unk>']
                            for word in words])
        Log.i('Loading %6d / %6d'%(index, data_len+valid_data_len))

      for index, line in enumerate(valid_raw_data):
        cols = line.split(',', 3)
        if do_cleaning:
          words = _clean_str(cols[3].strip('"')).split()
        else:
          words = cols[3].strip('"').split()
        self.max_len = max(self.max_len, len(words))
        # Tweet_id and authour ignore?
        if not has_lists:
          self.valid_label.append([cols[1]])
          self.valid_data.append(words)
        else:
          self.valid_label.append([self.sentiments[cols[1]]])
          self.valid_data.append([self.word_list[word] if word in self.word_list
                                  else self.word_list['<unk>']
                                  for word in words])
        Log.i('Loading %6d / %6d'%(index+data_len, data_len+valid_data_len))

      Log.i('Finish loading', True)

      Log.i('Start preprocessing...')

      if not has_lists:
        # Denoise by setting minimum freq
        self.word_list = ['<pad>', '<unk>'] + \
                         [key for key, value in self.word_counter.items() if value >= 3]

        # Save sentiment and word list
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

        # Convert to dict for fast searching
        self.sentiments = {word: index for index, word in enumerate(self.sentiments)}
        self.word_list = {word: index for index, word in enumerate(self.word_list)}

        # Convert text to index
        for index, [data_ent, label_ent] in enumerate(zip(self.data, self.label)):
          # <unk> (index 0) if word not found
          self.data[index] = [self.word_list[word] if word in self.word_list
                              else self.word_list['<unk>']
                              for word in data_ent]
          self.label[index] = [self.sentiments[word] for word in label_ent]

        # Convert text to index
        for index, [data_ent, label_ent] in enumerate(zip(self.valid_data, self.valid_label)):
          # <unk> (index 0) if word not found
          self.valid_data[index] = [self.word_list[word] if word in self.word_list
                                    else self.word_list['<unk>']
                                    for word in data_ent]
          self.valid_label[index] = [self.sentiments[word] for word in label_ent]

      data_len_list = [len(line) for line in self.data]
      self.data_len_mean = np.mean(data_len_list)
      self.data_len_std = np.std(data_len_list)
      self.data = [entry+[0]*(self.max_len-len(entry)) for entry in self.data]
      self.valid_data = [entry+[0]*(self.max_len-len(entry)) for entry in self.valid_data]

      Log.i('Finish preprocessing', True)
  def get_valid_set(self):
    valid_dataset = Dataset(None, _empty=True)
    valid_dataset.data = self.valid_data
    valid_dataset.label = self.valid_label
    valid_dataset.max_len = self.max_len
    del self.valid_data
    del self.valid_label
    return valid_dataset
  def get_vocab_size(self):
    return len(self.word_list)
  def get_max_len(self):
    return self.max_len
  def get_num_class(self):
    return len(self.sentiments)
  def __getitem__(self, index):
    return T.LongTensor(self.data[index]), T.LongTensor(self.label[index])
  def __len__(self):
    return len(self.data)
