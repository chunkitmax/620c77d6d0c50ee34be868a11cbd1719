from train import Train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-me', '--max_epoch', default=5000, type=int,
                    help='Max epochs the network goes')
parser.add_argument('-emb', '--emb_len', default=300, type=int,
                    help='Embedding length')
parser.add_argument('-esh', '--early_stop_history', default=100, type=int,
                    help='Number of epochs for determining early stopping')
parser.add_argument('-s', '--save', default=False, action='store_true',
                    help='Save, best f1score state will be saved')
parser.add_argument('-v', '--verbose', default=1, type=int, help='Verbose level')
parser.add_argument('-BOW', '--BagOfWord', default=False, action='store_true',
                    help='Train Bag of Word NN model')
parser.add_argument('-CNN', '--ConvNet', default=False, action='store_true',
                    help='Train Covolutional NN model')
parser.add_argument('-tb', '--tensorboard', default=False, action='store_true',
                    help='Log with TensorBoard')
parser.add_argument('-g', '--gpu', default=False, action='store_true',
                    help='Train with GPU support')

Args = parser.parse_args()

def main():
  mode = None
  if Args.BagOfWord:
    mode = True
  elif Args.ConvNet:
    mode = False
  if mode is not None:
    trainer = Train(train_BOW=True, max_epoch=Args.max_epoch, embedding_len=Args.emb_len,
                    early_stopping_history_len=Args.early_stop_history, use_cuda=Args.gpu,
                    use_tensorboard=Args.tensorboard, verbose=Args.verbose,
                    save_best_model=Args.save)
    trainer.train()
  else:
    parser.print_help()

if __name__ == '__main__':
  main()
