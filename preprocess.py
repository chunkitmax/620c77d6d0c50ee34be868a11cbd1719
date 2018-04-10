'''
preprocess.py
data preprocessing
'''

import re

def _clean_str(string):
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
