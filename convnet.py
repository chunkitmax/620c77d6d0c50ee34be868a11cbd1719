'''
convnet.py
implement CNN
'''

import math

import torch as T


class ConvNet(T.nn.Module):
  def __init__(self, embedding_len, num_vocab, num_class, doc_len,
               lr=.01, momentum=.9, kernel_size=[1], num_filter=[3],
               drop_rate=.5, stride=1, pooling_type='max', use_bn=False,
               use_cuda=False):
    super(ConvNet, self).__init__()
    # member declaration
    self.embedding_len = embedding_len
    self.num_vocab = num_vocab
    self.num_class = num_class
    self.doc_len = doc_len
    self.lr = lr
    self.momentum = momentum
    self.kernel_size = kernel_size
    if len(num_filter) < len(kernel_size) and len(num_filter) == 1:
      self.num_filter = num_filter*len(kernel_size)
    elif len(num_filter) < len(kernel_size):
      raise AssertionError('Number of filter is either 1 or same with length of kernel_size')
    else:
      self.num_filter = num_filter
    self.drop_rate = drop_rate
    self.stride = stride
    if pooling_type is None:
      self.pool_initializer = None
    elif pooling_type == 'max':
      self.pool_initializer = T.nn.MaxPool2d
    elif pooling_type == 'avg':
      self.pool_initializer = T.nn.AvgPool2d
    else:
      raise AssertionError('Unknown pooling type')
    self.use_bn = use_bn
    self.use_cuda = use_cuda
    # build model
    self._build_model()
  def _build_model(self):
    def init_weight(m):
      if isinstance(m, T.nn.Conv2d):
        T.nn.init.xavier_uniform(m.weight)
      elif isinstance(m, T.nn.Linear):
        m.weight.data.normal_().mul_(T.FloatTensor([2/m.weight.data.size()[0]]).sqrt_())

    # Embedding
    self.Embedding = T.nn.Embedding(self.num_vocab, self.embedding_len, padding_idx=0)
    self.CNN = []
    self.Pooling = []
    self.LeakyReLU = []
    self.BatchNorm = []
    self.DropOut = []
    # Convolution + Pooling
    for i in range(len(self.kernel_size)):
      tmp_CNN = T.nn.Conv2d(self.stride, self.num_filter[i], (self.kernel_size[i], self.embedding_len), bias=False)
      tmp_CNN.apply(init_weight)
      self.CNN.append(tmp_CNN)
      self.LeakyReLU.append(T.nn.LeakyReLU())
      if self.pool_initializer is not None:
        self.Pooling.append(self.pool_initializer((self.doc_len-i, 1)))
      if self.use_bn:
        self.BatchNorm.append(T.nn.BatchNorm2d(self.num_filter[i]))
      self.DropOut.append(T.nn.Dropout2d(self.drop_rate))
    # Dense
    self.Fc = T.nn.Linear(sum(self.num_filter), self.num_class)
    # Loss
    self.Loss_fn = T.nn.CrossEntropyLoss()
    # GPU support
    if self.use_cuda:
      [CNN.cuda() for CNN in self.CNN]
      [MP.cuda() for MP in self.Pooling]
      [BN.cuda() for BN in self.BatchNorm]
      self.cuda()
    if self.momentum > 0.:
      self.optimizer = T.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                   lr=self.lr, momentum=self.momentum, nesterov=True)
    else:
      self.optimizer = T.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                   lr=self.lr, momentum=0., nesterov=False)
  def forward(self, inputs):
    batch_size = inputs.size()[0]
    embeddings = self.Embedding(inputs)
    embeddings.unsqueeze_(1)
    CNN_output = []
    for i in range(len(self.kernel_size)):
      tmp_output = self.LeakyReLU[i](self.CNN[i](embeddings))
      if self.pool_initializer is not None:
        tmp_output = self.Pooling[i](tmp_output)
      if self.use_bn:
        tmp_output = self.BatchNorm[i](tmp_output)
      tmp_output = self.DropOut[i](tmp_output)
      CNN_output.append(tmp_output)
    concat = T.cat(CNN_output, 1).view([batch_size, -1])
    output = self.Fc(concat)
    _, max_indice = T.max(output, dim=1)
    return output, max_indice
  def get_loss_fn(self):
    return self.Loss_fn
  def get_optimizer(self):
    return self.optimizer
