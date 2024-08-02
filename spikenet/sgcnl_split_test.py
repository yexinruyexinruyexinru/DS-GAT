#!/user/bin/env python3
# -*- coding: utf-8 -*-
import time
import dataset
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import *
from sklearn import metrics
import random
#载入数据
data=dataset.DBLP()
#定义网络架构
class Single_Net(torch.nn.Module):
    def __init__(self):
        super(Single_Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 64)
    def forward(self, x,edge_index):
        x=self.conv1(x,edge_index)
        x = F.relu(x)
        return x
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(torch.nn.Module):
    def __init__(self,T_step):
        super(Net, self).__init__()
        self.snn_list = nn.ModuleList()
        for i in range(T_step):
            self.snn_list.append(Single_Net())
        self.lin1=nn.Linear(64,32)
        self.lin2 = nn.Linear(32,16)
        self.lin3=nn.Linear(16,data.num_classes)
        self.drop1=nn.Dropout(0.1)

    def forward(self,data):
        data.x=data.x.to(device)
        for i in range(len(data.x)):
            if i==0:
                x=self.snn_list[i](data.x[i],data.edges[i])
            else:
                x=x+self.snn_list[i](data.x[i],data.edges[i])
        x = F.relu(x)

        x=self.drop1(x)
        x = self.lin1(x)
        x = F.relu(x)

        x=self.drop1(x)
        x = self.lin2(x)
        x = F.relu(x)

        x=self.drop1(x)
        x = self.lin3(x)
        return F.log_softmax(x, dim=1)

device=torch.device('cpu')
model = Net(len(data)).to(device)

# 存储特征的信息
data_label=data.y.to(device)
# 特征数目
num_features=data.num_features
# 图的边的信息
edge_index=data.edges[len(data)-1].to(device)
# 边的权重
edges_weight=(data.edges_weight/100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
index=[i for i in range (len(data_label))]
random.shuffle(index)
train_nodes=index[0:int(0.4*len(index))]
# val_nodes=index[int(0.4*len(index)):int(0.7*len(index))]
test_nodes=index[int(0.4*len(index)):len(index)]

#模型训练
model.train()
# 定义时间步
T=100
start=time.time()
for epoch in range(T+1):
    print(epoch)
    optimizer.zero_grad()
    out = model(data)
    print(out.shape)

    loss = F.nll_loss(out[train_nodes], data_label[train_nodes])   #损失仅仅计算的是训练集的损失
    # loss.backward(retain_graph=True)
    loss.backward()
    optimizer.step()
    end=time.time()

    correct=0
    out_index=torch.argmax(out,dim=1)
    for i in range(len(train_nodes)):
        if out_index[train_nodes[i]]==data_label[train_nodes[i]]:
            correct+=1
    print("训练集的准确率为 %.04f"%(correct/len(train_nodes)))



# 模型评估
correct=0
model.eval()
with torch.no_grad():
    out = model(data)
    out_index=torch.argmax(out,dim=1)
    for i in range(len(test_nodes)):
        if out_index[test_nodes[i]]==data_label[test_nodes[i]]:
            correct+=1
    metric_macro = metrics.f1_score(data_label[test_nodes], out_index[test_nodes], average='macro')
    metric_micro = metrics.f1_score(data_label[test_nodes], out_index[test_nodes], average='micro')
    print("测试集的准确率为 %.04f,macro为： %.04f; micro为 %.04f" % (correct / len(test_nodes),metric_macro,metric_micro))

end=time.time()
elapsed_time=end-start
print("需要的时间为 %.02f"%(elapsed_time))

