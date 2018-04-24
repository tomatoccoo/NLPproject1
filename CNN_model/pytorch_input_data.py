import numpy as np
import torch
import torch.utils.data as data
import gensim
import os

class SIZE():
    def __init__(self):
        self.embedding = 100
        self.art_sentence = 128
        self.abs_sentence = 8
        self.words = 13


class MyDataSet(data.DataLoader):

    def __init__(self, dir, file_list, size, wvmodel):

        files = []
        with open(file_list, 'r') as f:
            for line in f.readlines():
                files.append(line.strip('\n'))


        self.files = files
        self.lens = len(files)
        self.size = size
        self.wvmodel = wvmodel
        self.dir = dir

    def __getitem__(self, index):# 定义迭代

        x = np.zeros((self.size.embedding, self.size.art_sentence, self.size.words))
        y = np.zeros((self.size.embedding, self.size.abs_sentence, self.size.words))


        with open(os.path.join(self.dir,(self.files)[index]), 'r') as f:
            g = f.read().split('\n\n')
            f.close()

        if len(g)!=2: # 如果文件有错误
            return torch.from_numpy(x), torch.from_numpy(y)


        art = [sentence.split(' ') for sentence in g[0].split('\n')]
        abs = [sentence.split(' ') for sentence in g[1].split('\n')[:-1]]

        for i in range(min(self.size.art_sentence, len(art))):# 对文章赋值
            for j in range(self.size.words):
                if art[i][j] == '__':# 如果缺失数据用0补齐
                    x[:, i, j] = 0
                else:
                    try:
                        x[:, i, j] = self.wvmodel[art[i][j]]
                    except KeyError :
                        x[:, i, j] = 0
                        pass
        for i in range(min(self.size.abs_sentence, len(abs))):#对摘要赋值
            for j in range(self.size.words):
                if abs[i][j] == '__':# 如果缺失数据用0补齐
                    y[:, i, j] = 0
                else:
                    try:
                        y[:, i, j] = self.wvmodel[abs[i][j]]
                    except KeyError :
                        y[:, i, j] = 0
                        pass

        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return len(self.files)

def generate_name_file(dir, name):
    file_names = os.listdir(dir)
    out = []
    with open(name, 'w') as f:
        for fname in file_names:
            if os.path.splitext(fname)[1] == '.new':
                out.append(fname)
        f.write('\n'.join(out))


if __name__ == '__main__':
    w2vmodel = gensim.models.Word2Vec.load('./word2vec/cnn_word2vec')
    print(type(w2vmodel['mother']))
