# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]//Proceedings of the 20th
    ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 701-710.
    (http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)
"""
from gensim.models import Word2Vec

from ..walker import RandomWalker

# 定义DeepWalk类
# RandomWalk + Word2Vec
# 首先，构造初始图结构/如果有则直接导入
# 然后，采用RandomWalk构建词向量序列, 形成预料库
# 之后，借助Word2Vec中的Skip-Gram或者CBOW方法来构建上下文模型
class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = RandomWalker(
            graph, p=1, q=1, )
        # 构造语料库, num_walks条长度为walk_length的语句
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    # 在随机游走之后进行Word2Vec训练
    # 基础参数：
    # embed_size=128, window_size=5, workers=3, iter=5
    #
    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter
        print("正在学习GraphEmbedding.....")
        # print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("完成GraphEmbedding学习!!!!")
        # print("Learning embedding vectors done!")
        self.w2v_model = model
        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("模型尚未训练!!!")
            # print("model not train")
            return {}

        # 若模型已经完成训练, 则计算模型转码结果
        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings
