import gensim
import torch
import torch.utils.data as Data
from torch.autograd import Variable



from  pytorch_input_data import *
from mymodel import myCNN



def numpy_array_to_words(numpy_array, wvmodel, size):
    result = []
    for j in range(size.abs_sentence):
        for i in range(size.words):
            vec = numpy_array[:,j,i]
            word = wvmodel.similar_by_vector(vec)
            result.append(word[0][0])
    return result




w2vmodel = gensim.models.Word2Vec.load('./word2vec/dailymail_word2vec')
cnn = myCNN()

cnn.load_state_dict(torch.load('cnn_NLP_model'))

test_data_set =  MyDataSet(r'D:\pycharmProject\data\dailymail_test\validation_new', 'data_test_file_name.txt',
                           size=SIZE(), wvmodel=w2vmodel)
test_loader = Data.DataLoader(dataset=test_data_set,
                              batch_size=1)

for step, (test_x, test_y) in enumerate(test_loader):
    test_v_x, test_v_y = Variable(test_x), Variable(test_y)
    output = cnn(test_v_x)
    break
a = output.view(100,8,13).data.numpy()
print(numpy_array_to_words(a, w2vmodel, SIZE()))
