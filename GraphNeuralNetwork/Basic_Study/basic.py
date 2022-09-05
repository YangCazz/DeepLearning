# 一、图神经网络基础知识的学习
#--------------------------------
# 导入图标准库
import numpy as np
import pandas as pd
import networkx as nx
# 构建边
edges = pd.DataFrame()
edges['sources'] = [1,1,1,2,2,3,3,4,4,5,5,5] # 边的一端
edges['targets'] = [2,4,5,3,1,2,5,1,5,1,3,4] # 边的另一端
edges['weights'] = [1 for i in range(12)]    # 边的权重
print(edges)
# 构建图
G = nx.from_pandas_edgelist(edges, source='sources', target='targets', edge_attr='weights')
# 计算出度
print("出度:{}".format(nx.degree(G)))
# 联通分量
print("联通分量:{}".format(list(nx.connected_components(G))))
# 图直径
print("图直径:{}".format(nx.diameter(G)))
# 度中心性
print("度中心性:{}".format(nx.degree_centrality(G)))
# 特征向量中心性
print("特征向量中心性:{}".format(nx.eigenvector_centrality(G)))
# betweenness
print("betweenness:{}".format(nx.betweenness_centrality(G)))
# closeness
print("closeness:{}".format(nx.closeness_centrality(G)))
# pagerank
print("PageRank:{}".format(nx.pagerank(G)))
# HITS
print("Hits:{}".format(nx.hits(G)))