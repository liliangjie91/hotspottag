#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by lljzhiwang on 2018/12/7 
from numpy import *
from sklearn.datasets import load_iris  # import datasets

# load the dataset: iris
iris = load_iris()
samples = iris.data
# print samples
target = iris.target

# import the LogisticRegression
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()  # 使用类，参数全是默认的
classifier.fit(samples, target)  # 训练数据来学习，不需要返回值

x = classifier.predict([5, 3, 5, 2.5])  # 测试数据，分类返回标记

print x

# 其实导入的是sklearn.linear_model的一个类：LogisticRegression， 它里面有许多方法
# 常用的方法是fit（训练分类模型）、predict（预测测试样本的标记）

# 不过里面没有返回LR模型中学习到的权重向量w，感觉这是一个缺陷