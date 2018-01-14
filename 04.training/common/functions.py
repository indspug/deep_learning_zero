# coding: utf-8

import numpy as np

##################################################
# シグモイド関数
##################################################
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

##################################################
# クロスエントロピー
##################################################
def cross_entropy(y, t):
	
	delta = 1e-7
	
	# yが1次元の場合は、2次元データに変換
	if y.ndim == 1:
		y = y.reshape(1, y.size)
		t = t.reshape(1, t.size)
	
	# 教師データがラベルの場合、
	# one-hot-vectorに変換
	if t.size != y.size:
		t1 = np.zeros_like(y)
		for i in range(t.shape[0]):
			maxidx = t[i]
			t1[i,maxidx] = 1
		t = t1
	
	n = y.shape[0]
	loss = -np.sum(t * np.log(y+delta))
	
	return (loss / n)

##################################################
# ソフトマックス関数
##################################################
def softmax(x):
	
	# そのまま計算するとオーバーフローするので
	# 最大値を引いた値でexpの計算を行う
	c = np.max(x)
	exp_a = np.exp(x - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	
	return y

