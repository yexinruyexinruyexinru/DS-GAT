#!/user/bin/env python3
# -*- coding: utf-8 -*-
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
from snn_utils import *
#载入数据
data=dataset.DBLP()
#定义网络架构


class SGC(nn.Module):
    def __init__(self, nfeat, nclass, bias=False):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass, bias=bias)
        # 初始化方法中服从均匀均匀分布
        torch.nn.init.xavier_normal_(self.W.weight)

    def forward(self, x):
        out = self.W(x)
        return out

class Single_Net(torch.nn.Module):
    def __init__(self):
        super(Single_Net, self).__init__()
        # self.conv1 = SGC(data.num_features, 64)
        self.conv1 = GCNConv(data.num_features, 64)
    def forward(self, x,edge_index):
        x=self.conv1(x,edge_index)
        # x = F.relu(x)
        return x

class Net(torch.nn.Module):
    def __init__(self,T_step):
        super(Net, self).__init__()
        self.snn_list = nn.ModuleList()
        for i in range(T_step):
            self.snn_list.append(Single_Net())
            # self.snn_list.append(SGC())
        # self.conv1 = GCNConv(data.num_features, 64)
        self.conv1=SGC(data.num_features, 64)
        self.lin1=nn.Linear(64,32)
        self.lin2 = nn.Linear(32,16)
        self.lin3=nn.Linear(16,data.num_classes)
        self.relu1=nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop1=nn.Dropout(0.12)
        # self.drop2 = nn.Dropout(0.8)
        # self.drop3 = nn.Dropout(0.5)
    def forward(self,data):
        for i in range(1):
            if i==0:
                x=self.snn_list[i](data.x[len(data.x)-i-1],data.edges[len(data.x)-i-1])
            else:
                x=x+0.02*self.snn_list[i](data.x[len(data.x)-i-1],data.edges[len(data.x)-i-1])
        # x=self.conv1(x,edge_index,edges_weight)
        x = self.relu1(x)

        x=self.drop1(x)
        x = self.lin1(x)
        x = self.relu2(x)

        x=self.drop1(x)
        x = self.lin2(x)
        x = self.relu3(x)

        x=self.drop1(x)
        x = self.lin3(x)
        return F.log_softmax(x, dim=1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
model = Net(len(data.x)).to(device)


# 存储特征的信息
data_label=data.y
# 特征数目
num_features=data.num_features
# 目前存储的只是特征信息
data_features=data.x[len(data)-1].to(device)
# 存储特征的信息
data_label=data.y.to(device)
# 图的边的信息
edge_index=data.edges[len(data)-1].to(device)
# 边的权重
edges_weight=(data.edges_weight/100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
index=[i for i in range (len(data_label))]
random.shuffle(index)
train_nodes=index[0:int(0.4*len(index))]
# val_nodes=index[int(0.4*len(index)):int(0.7*len(index))]
test_nodes=index[int(0.4*len(index)):len(index)]
print(data_features.size())

#模型训练
model.train()
# 定义时间步
T=200

for epoch in range(T+1):
    optimizer.zero_grad()
    out = model(data)    #模型的输入有节点特征还有边特征,使用的是全部数据
    # print(out.shape)
    # loss = nn.CrossEntropyLoss()
    # loss=loss(out[train_nodes], data_label[train_nodes])
    loss = F.nll_loss(out[train_nodes], data_label[train_nodes])   #损失仅仅计算的是训练集的损失
    loss.backward(retain_graph=True)
    optimizer.step()
    # end=time.time()

    correct=0
    out_index=torch.argmax(out,dim=1)
    for i in range(len(train_nodes)):
        if out_index[train_nodes[i]]==data_label[train_nodes[i]]:
            correct+=1
    print("%d 训练集的准确率为 %.04f"%(epoch,correct/len(train_nodes)))



# 模型评估
start=time.time()
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
print("GCN的运行时间为 %.02f"%(elapsed_time))

presim_len=4
sim_len=32
model = replace_activation_by_floor(model, 16)
# replace_activation_by_MPLayer(model, presim_len=presim_len, sim_len=sim_len)


# 模型评估
s1=time.time()
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
e1=time.time()
print("QCFS的运行时间为 %.02f"%(e1-s1))
# end=time.time()
# elapsed_time=end-start
# print("需要的时间为 %.02f"%(elapsed_time))

replace_activation_by_MPLayer(model, presim_len=presim_len, sim_len=sim_len)
s2=time.time()
correct=0
model.eval()
with torch.no_grad():
    for i in range(sim_len):
        if i==0:
            out = model(data)
        else:
            out+=model(data)
    out_index=torch.argmax(out,dim=1)
    for i in range(len(test_nodes)):
        if out_index[test_nodes[i]]==data_label[test_nodes[i]]:
            correct+=1
    metric_macro = metrics.f1_score(data_label[test_nodes], out_index[test_nodes], average='macro')
    metric_micro = metrics.f1_score(data_label[test_nodes], out_index[test_nodes], average='micro')
    print("测试集的准确率为 %.04f,macro为： %.04f; micro为 %.04f" % (correct / len(test_nodes),metric_macro,metric_micro))
e2=time.time()
print("SRP的运行时间为 %.02f"%(e2-s2))
