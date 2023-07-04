import re
from docx import Document
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

file_path = r"E:\预测数据\预测数据.docx"
doc = Document(file_path)
data = []
for p in doc.paragraphs:
    num_list = re.findall(r'\d+', p.text)
    print(num_list)
    if num_list != []:
        data.append(num_list)
data = np.array(data).astype(int)
data = data[::-1].copy()
print(data)
y1_np = data[:, 1:6]
y2_np = data[:, 6:]
x_np = data[:, 0]
# print(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

x = torch.from_numpy(x_np).to(device)
y1 = torch.from_numpy(y1_np).to(device)
y2 = torch.from_numpy(y2_np).to(device)


# 定义数据类
class dataset(Dataset):
    def __init__(self, data_feature, data_label):
        self.feature = data_feature
        self.label = data_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        feature = self.feature[item]
        label = self.label[item]
        return feature, label


# 从自己定义的数据集类当中加载数据
train_size = int(len(x)*0.8) # 必须是整数

train_x = x[:train_size+1]
train_y1 = y1[:train_size+1]
train_y2 = y2[:train_size+1]
test_x = x[train_size+1:]
test_y1 = y1[train_size+1:]
test_y2 = y2[train_size+1:]

train_data1 = TensorDataset(train_x, train_y1)
test_data1 = TensorDataset(test_x, test_y1)

train_data2 = TensorDataset(train_x, train_y2)
test_data2 = TensorDataset(test_x, test_y2)

train_loader1 = DataLoader(train_data1, batch_size=6, shuffle=False) # 为迭代器，每次迭代的形状都为(10, 10),所有样本一共有8次迭代
test_loader1 = DataLoader(test_data1, batch_size=6, shuffle=False) # num_batch = len(dataloader) # 所有样本能分为多少迭代次数或分为多少批次

train_loader2 = DataLoader(train_data2, batch_size=6, shuffle=False) # 为迭代器，每次迭代的形状都为(10, 10),所有样本一共有8次迭代
test_loader2 = DataLoader(test_data2, batch_size=6, shuffle=False) # num_batch = len(dataloader) # 所有样本能分为多少迭代次数或分为多少批次

if __name__ == '__main__':
    for batch, (x, y) in enumerate(train_loader1):
        print(f'\nthe {batch + 1} batch')
        print(f'the feature {x}')
        print(f'the label {y}')
        x = x.float()
        y = y.float()
        x = x[None, :, None]
        y = y[None, :, None]
        print(f'x is {x.shape}')
    pass
    for x, y in test_loader1:
        # print(f'\nthe {batch + 1} batch of test')
        print(f'\nthe test feature {x}')
        print(f'the label {y}')

