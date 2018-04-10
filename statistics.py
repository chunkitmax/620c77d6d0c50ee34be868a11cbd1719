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
  print('Number of sentences: %d'%(len(dataset.data)))
  print('Number of words: %d'%(sum(dataset.word_counter.values())))
  print('Number of unique words: %d(w/ min freq) %d(w/o min freq)'
        %(len(dataset.word_list), len(dataset.word_counter)))
  print('Coverage of your limited vocabulary: %.2f%%'
        %(sum([x for x in dataset.word_counter.values() if x<3])/len(dataset.word_counter)*100.))
  print('Top 10 most frequent words: ', dataset.word_counter.most_common(10))
  sentence_len = [len(line) for line in dataset.data]
  print('Maximum sentence length: %d'%(np.max(sentence_len)))
  print('Average sentence length: %.2f'%(np.mean(sentence_len)))
  print('Sentence length variation: %.2f'%(np.std(sentence_len)))
  class_counter = Counter([x[0] for x in dataset.label])
  sentiment_list = list(dataset.sentiments.keys())
  class_counter = {sentiment_list[key]: '%.2f%%'%(value / len(dataset.data)*100.)
                   for key, value in dict(class_counter).items()}

  print('Distribution of classes: ', class_counter)
