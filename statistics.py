'''
statistics.py
print out all data statistics
'''

import argparse
from collections import Counter
from zipfile import ZipFile

import numpy as np

from dataset import Dataset

argparser = argparse.ArgumentParser()
argparser.add_argument('-c', '--clean', default=False, action='store_true',
                       help='Do data cleaning')

if __name__ == '__main__':
  zf = ZipFile('hw3_dataset.zip', 'r')
  dataset = Dataset(zf.read('train.csv').decode('utf-8'), do_cleaning=argparser.parse_args().clean)
  valid_dataset = dataset.get_valid_set()

  word_counter = dataset.word_counter
  data = dataset.data + valid_dataset.data
  label = dataset.label + valid_dataset.label

  print('Number of sentences: %d'%(len(data)))
  print('Number of words: %d'%(sum(word_counter.values())))
  print('Number of unique words: %d(w/ min freq) %d(w/o min freq)'
        %(len(dataset.word_list), len(word_counter)))
  unk_rate = sum([x for x in word_counter.values() if x < 3])/ \
             len(word_counter)*100.
  print('Coverage of your limited vocabulary: %.2f%%, UNK token rate: %.2f%%'
        %(100.-unk_rate, unk_rate))
  print('Top 10 most frequent words: ', word_counter.most_common(10))
  print('Maximum sentence length: %d'%(dataset.max_len))
  print('Average sentence length: %.2f'%(dataset.data_len_mean))
  print('Sentence length variation: %.2f'%(dataset.data_len_std))
  class_counter = Counter([x[0] for x in label])
  sentiment_list = list(dataset.sentiments.keys())
  class_counter = {sentiment_list[key]: '%.2f%%'%(value / len(data)*100.)
                   for key, value in dict(class_counter).items()}

  print('Distribution of classes: ', class_counter)
