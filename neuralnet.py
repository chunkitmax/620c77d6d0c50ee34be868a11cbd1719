'''
neuralnet.py
implement BOW
'''

import math
import sys

import torch as T


class NeuralNet(T.nn.Module):
  def __init__(self, embedding_len, num_vocab, num_class, doc_len, num_hidden_layer=1,
               drop_rate=0.5, fc_dim=[2], activation_fn=[T.nn.functional.tanh],
               lr=.01, momentum=.9, use_cuda=False):
    super(NeuralNet, self).__init__()
    # member declaration
    self.embedding_len = embedding_len
    self.num_vocab = num_vocab
    self.num_class = num_class
    self.num_hidden_layer = num_hidden_layer
    self.drop_rate = drop_rate
    if len(fc_dim) < self.num_hidden_layer and len(fc_dim) > 1:
      raise AssertionError('Number of hidden layer should be either 1'
                           +' or number of hidden layers').with_traceback(sys.exc_info()[2])
    elif len(fc_dim) == 1:
      # self.fc_dim = [embedding_len*doc_len]+fc_dim*num_hidden_layer+[num_class]
      self.fc_dim = [embedding_len]+fc_dim*num_hidden_layer+[num_class]
    else:
      # self.fc_dim = [embedding_len*doc_len]+fc_dim+[num_class]
      self.fc_dim = [embedding_len]+fc_dim+[num_class]
    if len(activation_fn) < self.num_hidden_layer and len(activation_fn) > 1:
      raise AssertionError('Number of activation function should be either 1'
                           +' or number of layers').with_traceback(sys.exc_info()[2])
    elif len(activation_fn) == 1:
      self.activation_fn = activation_fn * num_hidden_layer
    else:
      self.activation_fn = activation_fn
    self.lr = lr
    self.momentum = momentum
    self.use_cuda = use_cuda
    # build model
    self._build_model()
  def _build_model(self):
    def init_weight(m):
      if isinstance(m, T.nn.Linear):
        m.weight.data.normal_().mul_(T.FloatTensor([2/m.weight.data.size()[0]]).sqrt_())

    # Embedding
    self.Embedding = T.nn.Embedding(self.num_vocab, self.embedding_len, padding_idx=0)
    # Dense
    self.Fc = []
    self.Dropout = []
    for i in range(len(self.fc_dim) - 1):
      if i > 0 and i + 1 < len(self.fc_dim):
        self.Dropout.append(T.nn.Dropout(self.drop_rate))
      tmp_layer = T.nn.Linear(self.fc_dim[i], self.fc_dim[i+1], bias=(i == 0))
      tmp_layer.apply(init_weight)
      self.Fc.append(tmp_layer)
    # Loss
    self.Loss_fn = T.nn.CrossEntropyLoss()

    # GPU support
    if self.use_cuda:
      for fc in self.Fc:
        fc.cuda()
      self.cuda()
    if self.momentum > 0.:
      self.optimizer = T.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                   lr=self.lr, momentum=self.momentum, nesterov=True)
    else:
      self.optimizer = T.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                   lr=self.lr, momentum=0., nesterov=False)
  def forward(self, inputs):
    output = self.Embedding(inputs)
    # batch_size = output.size()[0]
    # output = output.view([batch_size, -1])
    output = output.mean(dim=1)
    for i in range(len(self.fc_dim) - 1):
      if i > 0 and i + 1 < len(self.fc_dim):
        output = self.Dropout[i-1](output)
      output = self.Fc[i](output)
      if i < len(self.activation_fn):
        output = self.activation_fn[i](output)
    _, max_indice = T.max(output, dim=1)
    return output, max_indice
  def get_loss_fn(self):
    return self.Loss_fn
  def get_optimizer(self):
    return self.optimizer
