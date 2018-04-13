'''
convnet.py
implement CNN
'''

import math

import torch as T


class ConvNet(T.nn.Module):
  def __init__(self, embedding_len, num_vocab, num_class, doc_len,
               lr=.01, momentum=.9, ngrams=1, drop_rate=.5,
               num_filter=3, stride=1, use_bn=False, use_cuda=False):
    super(ConvNet, self).__init__()
    # member declaration
    self.embedding_len = embedding_len
    self.num_vocab = num_vocab
    self.num_class = num_class
    self.doc_len = doc_len
    self.lr = lr
    self.momentum = momentum
    self.ngrams = ngrams
    self.drop_rate = drop_rate
    self.num_filter = num_filter
    self.stride = stride
    self.use_bn = use_bn
    self.use_cuda = use_cuda
    # build model
    self._build_model()
  def _build_model(self):
    def init_weight(m):
      if isinstance(m, T.nn.Conv2d):
        s = m.kernel_size
        m.weight.data.normal_(0, math.sqrt(2. / (s[0] * s[1] * m.out_channels)))
      elif isinstance(m, T.nn.Linear):
        m.weight.data.normal_().mul_(T.FloatTensor([2/m.weight.data.size()[0]]).sqrt_())

    # Embedding
    self.Embedding = T.nn.Embedding(self.num_vocab, self.embedding_len)
    self.CNN_ngrams = []
    self.MaxPool = []
    self.LeakyReLU = []
    self.BatchNorm = []
    self.DropOut = []
    # Convolution + MaxPool
    for i in range(self.ngrams):
      tmp_CNN = T.nn.Conv2d(self.stride, self.num_filter, (i+1, self.embedding_len), bias=False)
      tmp_CNN.apply(init_weight)
      self.CNN_ngrams.append(tmp_CNN)
      self.LeakyReLU.append(T.nn.LeakyReLU())
      self.MaxPool.append(T.nn.MaxPool2d((self.doc_len-i, i+1)))
      self.DropOut.append(T.nn.Dropout2d(self.drop_rate))
      if self.use_bn:
        self.BatchNorm.append(T.nn.BatchNorm2d(self.num_filter))
    # Dense
    self.Fc = T.nn.Linear(self.ngrams*self.num_filter, self.num_class)
    # Loss
    self.Loss_fn = T.nn.CrossEntropyLoss()
    # GPU support
    if self.use_cuda:
      [CNN.cuda() for CNN in self.CNN_ngrams]
      [MP.cuda() for MP in self.MaxPool]
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
    ngram_output = []
    for i in range(self.ngrams):
      tmp_output = self.MaxPool[i](self.LeakyReLU[i](self.CNN_ngrams[i](embeddings)))
      tmp_output = self.DropOut[i](tmp_output)
      if self.use_bn:
        tmp_output = self.BatchNorm[i](tmp_output)
      ngram_output.append(tmp_output)
    concat = T.cat(ngram_output, 1).view([batch_size, -1])
    output = self.Fc(concat)
    return output, T.max(output, dim=1)
  def get_loss_fn(self):
    return self.Loss_fn
  def get_optimizer(self):
    return self.optimizer
