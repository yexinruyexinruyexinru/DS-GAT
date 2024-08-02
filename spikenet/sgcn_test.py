#!/user/bin/env python3
# -*- coding: utf-8 -*-
#!/user/bin/env python3
# -*- coding: utf-8 -*-
import time
import dataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import *
#载入数据
data=dataset.DBLP()
# 存储的邻接矩阵的信息
data_adj=data.adj[0]
# 目前存储的只是特征信息
data_features=data.x[0]
# 存储特征的信息
data_label=data.y
# 图的边的信息
edge_index=data.edges
# 特征数目
num_features=data.num_features
#定义网络架构
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)  #输入=节点特征维度，16是中间隐藏神经元个数
        # self.sneuron1=neuron.LIFNode(tau=80, v_threshold=0.66, v_reset=0.11,detach_reset=True)
        self.conv2 = GCNConv(16, data.num_classes)
        # self.sneuron2 = neuron.LIFNode(tau=80, v_threshold=0.5, v_reset=0.1, surrogate_function=surrogate.Sigmoid(),detach_reset=True)
    def forward(self, x, edge_index,edges_weight):
        x = self.conv1(x, edge_index,edges_weight)
        # x=self.sneuron1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index,edges_weight)
        # x=self.sneuron2(x)
        # return x
        return F.log_softmax(x, dim=1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
model = Net().to(device)

# 目前存储的只是特征信息
data_features=data.x[26].to(device)
print(type(data_features))
# 存储特征的信息
data_label=data.y.to(device)
print(type(data_label))
# 图的边的信息
edge_index=data.edges[26].to(device)
# 边的权重
edges_weight=(data.edges_weight/100)
print(edges_weight)
# print(edge_index)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
#模型训练
model.train()
# 定义时间步
T=500
for epoch in range(T+1):
    start=time.time()
    optimizer.zero_grad()
    out = model(data_features, edge_index,edges_weight)    #模型的输入有节点特征还有边特征,使用的是全部数据
    print(out.shape)
    # print(out)
    loss = F.nll_loss(out, data_label)   #损失仅仅计算的是训练集的损失
    loss.backward(retain_graph=True)
    optimizer.step()
    end=time.time()
    elapsed_time=end-start
    correct=0
    out_index=torch.argmax(out,dim=1)
    for i in range(len(data_label)):
        if out_index[i]==data_label[i]:
            correct+=1
    print("训练集的准确率为 %.04f"%(correct/len(data_label)))
    print("epoch %d 需要的时间为 %.02f"%(epoch,elapsed_time))
