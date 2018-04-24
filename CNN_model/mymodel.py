import torch.nn as nn

class myCNN(nn.Module):
    def __init__(self):

        super(myCNN, self).__init__()
        self.conv1 = nn.Sequential(
            # 对输入卷积
            nn.Conv2d(in_channels=100, out_channels=16, kernel_size=5, stride=1, padding=2).double(),
            # 经过relu
            nn.ReLU(),
            # 池化
            nn.MaxPool2d(kernel_size=(4,1))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 5, 1, 2).double(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 5, 1, 2).double(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 1))
        )
        self.out1 = nn.Linear(16*8*13, 8*8*13).double()
        self.out2 = nn.Linear(8*8*13, 100*8*13).double()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.out1(x)
        output = self.out2(x)
        return output