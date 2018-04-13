'''
train.py
train model
'''

import gzip
import sys
from collections import deque
from zipfile import ZipFile

import numpy as np
import torch as T
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from convnet import ConvNet
from dataset import Dataset
from log import Logger
from neuralnet import NeuralNet


class Train:
  def __init__(self, train_BOW=True, max_epoch=5000, target_file='hw3_dataset.zip',
               batch_size=100, embedding_len=300, use_cuda=False,
               use_tensorboard=False, early_stopping_history_len=100,
               early_stopping_allowance=5, verbose=1, save_best_model=False):
    self.logger = Logger(verbose_level=verbose)
    self.train_BOW = train_BOW
    self.max_epoch = max_epoch
    self.target_file = target_file
    self.batch_size = batch_size
    self.embedding_len = embedding_len
    self.use_cuda = use_cuda
    self.use_tensorboard = use_tensorboard
    self.early_stopping_history_len = early_stopping_history_len
    self.early_stopping_allowance = early_stopping_allowance
    self.save_best_model = save_best_model
    self.counter = 0
  def train(self):
    if self.train_BOW:
      lr = [0.01, 0.001, 0.0001]
      drop_rate = [0.1, 0.3, 0.5]
      hidden_size = [128, 256, 512]
      num_hidden_layer = [1, 2, 3]
      def train_net(lr_i, dr_i, hs_i, nhl_i):
        train_data_loader, valid_data_loader,\
        vocab_size, max_len, num_class = self._get_data_loader()
        tmp_net = NeuralNet(self.embedding_len, vocab_size, num_class,
                            max_len, num_hidden_layer[nhl_i], drop_rate[dr_i],
                            fc_dim=[hidden_size[hs_i]], lr=lr[lr_i], momentum=.9,
                            use_cuda=self.use_cuda)
        loss = self._train(tmp_net, train_data_loader, valid_data_loader,
                           'BOW_%d_%d_%d_%d'%(lr_i, dr_i, hidden_size[hs_i],
                                              num_hidden_layer[nhl_i]))
        del train_data_loader, valid_data_loader, tmp_net
        return loss
      prev_loss = 999.
      prev_i, prev_j = -1, -1
      # Grid search
      # seach for optimal num_hidden_layer and hidden_size
      for j in range(3):
        for i in range(3):
          loss = train_net(0, 0, i, j)
          if loss >= prev_loss:
            break
          prev_loss = loss
          prev_i, prev_j = i, j
      # search for optimal lr
      prev_k = 0
      for k in range(2, 0, -1):
        loss = train_net(k, 0, prev_i, prev_j)
        if loss > prev_loss:
          break
        prev_loss = loss
        prev_k = k
      # search for optimal drop rate
      for l in range(2, 0, -1):
        loss = train_net(prev_k, l, prev_i, prev_j)
        if loss > prev_loss:
          break
        prev_loss = loss
    else:
      lr = [0.1, 0.01, 0.001]
      ngrams = [1, 2, 3]
      drop_rate = [0.1, 0.3, 0.5]
      num_filter = [3, 4, 5]
      stride = [1, 1, 2]
      for i in range(3):
        train_data_loader, valid_data_loader, \
        vocab_size, max_len, num_class = self._get_data_loader()
        tmp_net = ConvNet(self.embedding_len, vocab_size, num_class,
                          max_len, drop_rate=drop_rate[i], ngrams=ngrams[i],
                          num_filter=num_filter[i], stride=stride[i],
                          use_bn=True, lr=lr[i], momentum=.9,
                          use_cuda=self.use_cuda)
        self._train(tmp_net, train_data_loader, valid_data_loader, 'CNN_'+str(i))
  def _get_data_loader(self, train_valid_ratio=.2):
    zf = ZipFile(self.target_file, 'r')
    data = zf.read('train.csv').decode('utf-8').strip().split('\n')[1:]
    whole_dataset = Dataset(data, train_valid_ratio)
    valid_dataset = whole_dataset.get_valid_set()
    return DataLoader(whole_dataset, batch_size=self.batch_size, shuffle=True), \
           DataLoader(valid_dataset, batch_size=self.batch_size), \
           whole_dataset.get_vocab_size(), \
           whole_dataset.get_max_len(), \
           whole_dataset.get_num_class()
  def _train(self, net, train_data_loader, valid_data_loader, identity=None):
    if identity is None:
      identity = 'Net'+str(self.counter)
      self.counter += 1
    if self.use_tensorboard:
      from tensorboardX import SummaryWriter
      self.writer = SummaryWriter(identity+'_logs')
    optimizer = net.get_optimizer()
    loss_fn = net.get_loss_fn()
    self.logger.i('Start training %s...'%(identity), True)
    try:
      total_batch_per_epoch = len(train_data_loader)
      loss_history = deque(maxlen=self.early_stopping_history_len)
      max_fscore = 0.
      best_val_loss = 999.
      early_stopping_violate_counter = 0
      epoch_index = 0
      for epoch_index in range(self.max_epoch):
        losses = 0.
        acc = 0.
        counter = 0
        self.logger.i('[ %d / %d ] epoch:'%(epoch_index + 1, self.max_epoch), True)
        net.train()
        for batch_index, (data, label) in enumerate(train_data_loader):
          data = T.autograd.Variable(data)
          label = T.autograd.Variable(label)
          if self.use_cuda:
            data = data.cuda()
            label = label.cuda()
          output, predicted = net(data)
          acc += (label.squeeze() == predicted).float().mean().data
          loss = loss_fn(output, label.view(-1))
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          losses += loss.data.cpu()[0]
          counter += 1
          progress = min((batch_index + 1) / total_batch_per_epoch * 20., 20.)
          self.logger.d('[%s] (%3.f%%) loss: %.4f, acc: %.4f'%
                        ('>'*int(progress)+'-'*(20-int(progress)), progress * 5.,
                         losses / counter, acc / counter))
        mean_loss = losses / counter
        valid_losses = 0.
        valid_prec = 0.
        valid_recall = 0.
        valid_fscore = 0.
        valid_step = 0
        net.eval()
        for valid_step, (data, label) in enumerate(valid_data_loader):
          data = T.autograd.Variable(T.LongTensor(data))
          label = T.autograd.Variable(T.LongTensor(label))
          if self.use_cuda:
            data = data.cuda()
            label = label.cuda()
          output, predicted = net(data)
          prec, recall, \
          fscore, _ = precision_recall_fscore_support(label.data.tolist(),
                                                      predicted.data.tolist(),
                                                      average='weighted')
          valid_prec += prec
          valid_recall += recall
          valid_fscore += fscore
          valid_losses += loss_fn(output, label.view(-1)).data.cpu()[0]
        mean_val_loss = valid_losses/(valid_step+1)
        mean_fscore = valid_fscore/(valid_step+1)
        self.logger.d(' -- val_loss: %.4f, prec: %.4f, rec: %.4f, fscr: %.4f'%
                      (mean_val_loss, valid_prec/(valid_step+1),
                       valid_recall/(valid_step+1), mean_fscore),
                      reset_cursor=False)
        if self.use_tensorboard:
          self.writer.add_scalar('train_loss', mean_loss, epoch_index)
          self.writer.add_scalar('train_acc', acc / counter, epoch_index)
          self.writer.add_scalar('val_loss', mean_val_loss, epoch_index)
          self.writer.add_scalar('val_fscr', mean_fscore, epoch_index)
        loss_history.append(mean_val_loss)
        if mean_val_loss > np.mean(loss_history):
          early_stopping_violate_counter += 1
          if early_stopping_violate_counter >= self.early_stopping_allowance:
            self.logger.i('Early stopping...', True)
            break
        else:
          early_stopping_violate_counter = 0
        if self.save_best_model and mean_fscore > max_fscore:
          self._save(epoch_index, net, loss_history, mean_fscore, identity)
          max_fscore = mean_fscore
        if mean_val_loss < best_val_loss:
          best_val_loss = mean_val_loss
        self.logger.d('', True, False)
    except KeyboardInterrupt:
      self.logger.i('\n\nInterrupted', True)
    if self.use_tensorboard:
      self.writer.close()
    self.logger.i('Finish', True)
    return best_val_loss
  def _save(self, global_step, net, loss_history, best_fscore, identity):
    T.save({
        'epoch': global_step+1,
        'state_dict': net.state_dict(),
        'loss_history': loss_history,
        'best_fscore': best_fscore,
        'optimizer': net.optimizer.state_dict()
    }, identity+'_best')

if __name__ == '__main__':
  trainer = Train(use_cuda=True, use_tensorboard=True)
  trainer.train()
