#!/user/bin/env python3
# -*- coding: utf-8 -*-
#导入所需的包
import numpy as np
import pandas as pd
import torch
# 显示所有列
# pd.set_option('display.max_columns',None)
# # 显示所有行
# pd.set_option('display.max_rows',None)
# #导入npy文件路径位置
#
#
import numpy as np
# 读取 txt 文件
data = np.loadtxt("D:\A_data\\tmall.txt")
print(data.shape)
# 将数据保存为 npy 文件
# np.save("D:\A_data\\tmall.npy", data)


# test = np.load('D:\A_data\\tmall.npy')
# print(test.shape)
# # print(test[0])



# print(test.shape)
# print(test)
# print(type(test))
# print(test[:,1,:])

# 如何表示的图的结构的呢
# 三维的数据是如何表示图的结构的呢
# 28085条数据，每条数据有27个属性，128表示的是时间步吗

# a=torch.rand(4,1,28,28)
# print(a.shape)

# a=a.view(-1,4,28*28)
# print(a.shape)

# dict={'A':['a','b','c'],'B':['e','f','g']}
#
# print(type(dict))
# for key in dict:
#     value = zip(*dict[key])
#     print(dict[key])


import os.path as osp
from collections import defaultdict, namedtuple
'''
def _read_graph():
    filename = 'D:\Areproduce\SpikeNet-master\SpikeNet-master\data\dblp\dblp.txt'
    # 2023.6.15 这一行代码的作用是什么，key对应的value值的类型并不确定
    d = defaultdict(list)
    N = 0
    with open(filename) as f:
        for line in f:
            x, y, t = line.strip().split()
            x, y = int(x), int(y)
            d[t].append((x, y))
            N = max(N, x)
            N = max(N, y)
    N += 1
    edges = []
    index=0
    for time in sorted(d):
        print("第%d次循环"%(index))
        index+=1
        row, col = zip(*d[time])
        print("row:",row)
        print("col:",col)
        edge_now = np.vstack([row, col])
        print(edge_now)
        edges.append(edge_now)
    return edges, N

edges,N=_read_graph()
print(len(edges))
print(N)
'''

'''
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])
a=[1,2,3,6]
print(a)
lb.transform(a)
print(lb.transform(a))
print(a)
'''

# import numpy as np
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(5, 4))
# ax = plt.subplot(111)


# Function that runs the simulation
# tau: time constant (in ms)
# t0, t1, t2: time of three input spikes
# w: input synapse weight
# threshold: threshold value to produce a spike
# reset: reset value after a spike
# def LIF(tau=10, t0=20, t1=30, t2=35, w=0.8, threshold=1.0, reset=0.0):
#     # Spike times, keep sorted because it's more efficient to pop the last value off the list
#     times = [t0, t1, t2]
#     times.sort(reverse=True)
#     # set some default parameters
#     duration = 100  # total time in ms
#     dt = 0.1  # timestep in ms
#     alpha = np.exp(-dt / tau)  # this is the factor by which V decays each time step
#     V_rec = []  # list to record membrane potentials
#     V = 0.0  # initial membrane potential
#     T = np.arange(np.round(duration / dt)) * dt  # array of times
#     spikes = []  # list to store spike times
#     # run the simulation
#     # plot everything (T is repeated because we record V twice per loop)
#     ax.clear()
#     for t in times:
#         ax.axvline(t, ls=':', c='b')
#
#     for t in T:
#         V_rec.append(V)  # record
#         V *= alpha  # integrate equations
#         if times and t > times[-1]:  # if there has been an input spike
#             V += w
#             times.pop()  # remove that spike from list
#         V_rec.append(V)  # record V before the reset so we can see the spike
#         if V > threshold:  # if there should be an output spike
#             V = reset
#             spikes.append(t)
#     ax.plot(np.repeat(T, 2), V_rec, '-k', lw=2)
#     for t in spikes:
#         ax.axvline(t, ls='--', c='r')
#     ax.axhline(threshold, ls='--', c='g')
#     ax.set_xlim(0, duration)
#     ax.set_ylim(-1, 2)
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('Voltage')
#     plt.tight_layout()
#     plt.show()
#
#
# LIF()



# 查看node2label中的标签数目以及标号数目
'''
file=open("D:\Areproduce\SpikeNet-master\SpikeNet-master\data\\tmall\\node2label.txt")
filelines=file.readlines()
m=len(filelines)
print(m)
id=[]
label=[]
for i in range(m):
    lin=filelines[i].split(' ')
    id.append(int(lin[0]))
    label.append(lin[1])
# print(id)
# print(label)
id.sort()
print(len(set(id)))
print(len(set(label)))
# print(id)
'''


# 需要导入模块: from numpy.lib import format [as 别名]
# 或者: from numpy.lib.format import read_magic [as 别名]
from numpy.lib import format
from numpy.lib.format import read_magic
import io

# def test_read_magic():
#     s1 = io.BytesIO()
#     s2 = io.BytesIO()
#
#     arr = np.ones((3, 6), dtype=float)
#
#     format.write_array(s1, arr, version=(1, 0))
#     format.write_array(s2, arr, version=(2, 0))
#
#     s1.seek(0)
#     s2.seek(0)
#
#     version1 = format.read_magic(s1)
#     version2 = format.read_magic(s2)
#     print(version1)
#     print(version2)
#     assert_(version1 == (1, 0))
#     assert_(version2 == (2, 0))
#
#     assert_(s1.tell() == format.MAGIC_LEN)
#     assert_(s2.tell() == format.MAGIC_LEN)
#
# test_read_magic()