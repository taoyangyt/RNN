import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import RNN_construct
from load_data import train_loader1, test_loader1, train_loader2, test_loader2, test_x

# 定义一些超参数
INPUT_SIZE = 1
LR = 0.01
epoch = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 选择模型
model = RNN_construct.Rnn(INPUT_SIZE).to(device)
print(model)

# 定义优化器和损失函数
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

h_state = None  # 第一次的时候，暂存为0


def train_loop(train_dataloader, model, loss_function, optimizer, h_state):
    size = len(train_dataloader.dataset) # 所有的样本数
    for batch, (x, y) in enumerate(train_dataloader):
        # x = x.view(10, 1).float()
        # y = y.view(10, 1).float()
        x = x.float()
        y = y.float()
        x = x[None, :, None]
        y = y[None, :, :]
        pre, h_state = model(x, h_state)
        h_state = h_state.data

        loss = loss_function(pre, y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if (batch+1)%1 == 0:
            loss, current = loss.item(), (batch+1)*len(x)
            print(f'loss:{loss:>7f} [{current:>5f}/{size:>5d}]')


def test_loop(test_dataloader, model, loss_function, h_state):
    test_loss = 0 # 回归任务当中可以没有准确率
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.float()
            y = y.float()
            x = x[None, :, None]
            y = y[None, :, :]
            pre, h_state = model(x, h_state)
            loss = loss_function(pre, y)
            test_loss += loss.item()

    # test_loss /= batch_size*2
    # test_loss /= 2
    print(f'Test error:\n Avg loss:{test_loss:>8f} \n')


if __name__ == '__main__':
    plt.ion()
    plt.show()
    x_test = test_x.float()
    x_test = x_test[None, :, None]
    for i in range(epoch):
        print(f'\nEpoch {i+1}\n ---------------------')
        train_loop(train_loader1, model, loss_function, optimizer, h_state)
        test_loop(test_loader1, model, loss_function, h_state)
        pre, h_state = model(x_test, h_state)
        print(f'pre is {pre}')
        # plt.cla()
        # plt.scatter(x_np, y1_np)
        # plt.plot(x_np, pre)
        # plt.pause(0.1)
    torch.save(model.state_dict(), 'model1_weights.pth')
    print('Done')
    plt.ioff()
    plt.show()