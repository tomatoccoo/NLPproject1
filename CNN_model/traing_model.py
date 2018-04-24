import torch
from torch.autograd import Variable
import torch.utils.data as Data
import gensim
import torch.nn as nn



from  pytorch_input_data import *
from my_cnn_model import myCNN

EPOCH = 10
BATCH_SIZE = 10
LR = 0.1
embedding_size = 100

# 加载word2vec模型
w2vmodel = gensim.models.Word2Vec.load('./word2vec/dailymail_word2vec')

# 创建需要处理的文件目录


generate_name_file(r'D:\pycharmProject\data\dailymail_train\training_new', 'data_train_file_name.txt')
generate_name_file(r'D:\pycharmProject\data\dailymail_test\validation_new', 'data_test_file_name.txt')


# pytorch接口，文件载入
train_data_set = MyDataSet(r'D:\pycharmProject\data\dailymail_train\training_new', 'data_train_file_name.txt',
                           size=SIZE(), wvmodel=w2vmodel)
train_loader = Data.DataLoader(dataset=train_data_set,
                               batch_size=BATCH_SIZE,
                               shuffle=True) # shuffle 打乱

test_data_set =  MyDataSet(r'D:\pycharmProject\data\dailymail_test\validation_new', 'data_test_file_name.txt',
                           size=SIZE(), wvmodel=w2vmodel)
test_loader = Data.DataLoader(dataset=test_data_set,
                              batch_size=BATCH_SIZE)


cnn = myCNN()
if torch.cuda.is_available():
    cnn = myCNN().cuda()
print(cnn)

cnn.load_state_dict(torch.load('cnn_NLP_model'))


optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.MSELoss()

for epoch in range(EPOCH):



    train_loss = 0
    for step , (x, y) in enumerate(train_loader):


        # 前馈
        if torch.cuda.is_available():
            v_x = Variable(x.cuda())
            v_y = Variable(y.cuda())
        else:
            v_x = Variable(x)
            v_y = Variable(y)

        output = cnn(v_x)
        loss = loss_function(output, v_y)

        train_loss = train_loss+loss.data[0]
        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 检查错误率
        # 测试
        if step % 100 == 0:
            print('step:', step)

            test_loss = 0
            for step, (test_x, test_y) in enumerate(test_loader):
                if torch.cuda.is_available():
                    test_v_x, test_v_y = Variable(test_x.cuda()), Variable(test_y.cuda())
                else:
                    test_v_x, test_v_y = Variable(test_x), Variable(test_y)

                output = cnn(test_v_x)
                test_loss = test_loss+loss_function(output, test_v_y).data[0]
                break # 只求一个batch的错误率

            print('Test Loss:{:.6f}'.format(test_loss/BATCH_SIZE))
    print('train loss{:.6f}'.format(train_loss/train_data_set.__len__()))


torch.save(cnn.state_dict(), 'cnn_NLP_model')

