import os
import gensim
import re
import time

# 训练一个word2vec的模型
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            try:
                f = open(os.path.join(self.dirname, fname), 'r', encoding='UTF-8')
                g = f.read().split('\n\n')
                for line in g[1].split('\n'):
                    yield line[:-4].replace('@', '').split(' ')
                for line in g[2].split('\n'):
                    yield line[:-4].replace('@', '').split(' ')
            except:
                yield []

time_start = time.time()
sentences = MySentences(r'D:\school_study\cnn_dailymail\cnn_dailymail\dailymail\training')
model = gensim.models.Word2Vec(sentences)
time_end = time.time()
model.save('./wor2vec/dailymail_word2vec')
print(model['entity17'])

print(time_end-time_start)

