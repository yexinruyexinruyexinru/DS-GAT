#!/user/bin/env python3
# -*- coding: utf-8 -*-
#!/user/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
from sklearn import metrics
from torchsummary import summary
from tqdm import tqdm
from spikenet import dataset, neuron
from torch.utils.data import DataLoader
from spikenet.layers import SAGEAggregator
from spikenet.utils import (RandomWalkSampler, Sampler, add_selfloops,
                            set_seed, tab_printer)
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
#定义网络架构
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.x=x
        self.edge_index=edge_index
        self.conv1 = GCNConv(data.num_features, 16)  #输入=节点特征维度，16是中间隐藏神经元个数
        self.conv2 = GCNConv(16, data.num_classes)
    def forward(self,x,edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x,edge_index)
        return F.log_softmax(dim=1)


# 定义VGG网络
class VGG19(torch.nn.Module):
    def __init__(self,In_channel=1,classes=5):
        super(VGG19, self).__init__()
        self.feature = torch.nn.Sequential(

            torch.nn.Conv1d(In_channel, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.AdaptiveAvgPool1d(7)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(3584,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, classes),
        )

    def forward(self, x):
        # print(type(x))
        x=x.float()
        # print(type(x))
        x = self.feature(x)
        x = x.view(-1, 3584)
        x = self.classifer(x)
        return x

# 训练模型
# def train():
#     model.train()
#     for nodes in tqdm(train_loader, desc='Training'):
#         nodes=nodes.float()
#         l=len(nodes)
#         nodes=nodes[None,None,]
#         optimizer.zero_grad()
#         out=[]
#         nodes=nodes.reshape(l)
#         for i in nodes:
#             i=i.long()
#             node=data.x[0][i]
#             node=node[None,None,]
#             out.append(list(model(node).reshape(10)))
#         out=torch.Tensor(out)
#         out.requires_grad_(True)
#         loss_fn(out, y[nodes.long()].flatten()).backward()
#         optimizer.step()


# @torch.no_grad()
# def test(loader):
#     model.eval()
#     logits = []
#     labels = []
#     # print(loader)
#     for nodes in loader:
#         # print(nodes)
#         # print(model(nodes).shape)
#         # print(y[nodes])
#         # l=len(nodes)
#         # nodes=nodes[None,None,]
#         for i in nodes:
#             i = i.long()
#             node = data.x[0][i]
#             node = node[None, None,]
#             # print(node)
#             logits.append(list(model(node).reshape(10)))
#         labels.append(y[nodes])
#     # print(type(logits))
#     logits=torch.Tensor(logits)
#     # print(type(logits))
#     # print(logits)
#     # logits = torch.cat(logits, dim=0).cpu()
#     labels = torch.cat(labels, dim=0).cpu()
#     # for i in range(8427):
#     #     print(logits[i])
#     # argmax()导出相关维度上最大值的索引
#     logits = logits.argmax(1)
#     # print(logits.shape)
#     # print(logits)
#     # print(labels.shape)
#     metric_macro = metrics.f1_score(labels, logits, average='macro')
#     metric_micro = metrics.f1_score(labels, logits, average='micro')
#     return metric_macro, metric_micro

def train():
    model.train()
    for nodes in tqdm(train_loader, desc='Training'):
        x=data.x[0][nodes]
        edge_index=data.edges[0]
        optimizer.zero_grad()
        out=model(x, edge_index)
        loss_fn(model(nodes), y[nodes]).backward()
        optimizer.step()
@torch.no_grad()
def test(loader):
    model.eval()
    logits = []
    labels = []
    for nodes in loader:
        logits.append(model(nodes))
        labels.append(y[nodes])
    logits = torch.cat(logits, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    # argmax()导出相关维度上最大值的索引
    logits = logits.argmax(1)
    metric_macro = metrics.f1_score(labels, logits, average='macro')
    metric_micro = metrics.f1_score(labels, logits, average='micro')
    return metric_macro, metric_micro

if __name__ == '__main__':
    # 加载数据
    data = dataset.DBLP()
    # 划分数据
    data.split_nodes(train_size=0.4, val_size=0.3,
                     test_size=0.3, random_state=True)
    train_loader = DataLoader(data.train_nodes.tolist(), pin_memory=False, batch_size=1024, shuffle=True)
    val_loader = DataLoader(data.test_nodes.tolist() if data.val_nodes is None else data.val_nodes.tolist(),
                            pin_memory=False, batch_size=200000, shuffle=False)
    test_loader = DataLoader(data.test_nodes.tolist(), pin_memory=False, batch_size=200000, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    y = data.y.to(device)
    x=data.x[0].to(device)
    edge_index=data.edges[0].to(device)
    # print(type(train_loader))
    # 定义模型
    # model = VGG19(In_channel=1,classes=10).to(device)
    model=Net().to(device)
    # 定义优化器以及损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_metric = test_metric = 0
    start = time.time()
    for epoch in range(1, 101):
        train()
        val_metric, test_metric = test(val_loader), test(test_loader)
        if val_metric[1] > best_val_metric:
            best_val_metric = val_metric[1]
            best_test_metric = test_metric
        end = time.time()
        print(
         f'Epoch: {epoch:03d}, Val: {val_metric[1]:.4f}, Test: {test_metric[1]:.4f}, Best: Macro-{best_test_metric[0]:.4f}, Micro-{best_test_metric[1]:.4f}, Time elapsed {end-start:.2f}s')